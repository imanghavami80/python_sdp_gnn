"""Conservative source-level Java receiver resolution; no whole-program points-to analysis."""

from dataclasses import dataclass, field, replace
from typing import Any

import javalang.tree as jt
from javalang.ast import Node


def declared_name(type_node: Any) -> str | None:
    """Use the outer declared type, never a generic argument as its receiver."""
    if type_node is None or isinstance(type_node, jt.BasicType):
        return None
    if getattr(type_node, "dimensions", None):
        return None  # an array receiver is not an instance of its component type
    parts = []
    while type_node is not None:
        parts.append(type_node.name)
        type_node = getattr(type_node, "sub_type", None)
    return ".".join(parts)


def members(node):
    body = getattr(node, "body", None)
    return body.declarations if isinstance(body, jt.EnumBody) else (body or [])


def executables(node):
    return [m for m in members(node) if isinstance(m, (jt.MethodDeclaration, jt.ConstructorDeclaration))]


@dataclass
class TypeInfo:
    name: str
    file: str
    node: Node
    resolver: Any
    parents: list[str] = field(default_factory=list)
    fields: dict[str, str | None] = field(default_factory=dict)
    methods: dict[tuple[str, int], list[tuple[str | None, bool, bool]]] = field(default_factory=dict)


class ProjectTypes:
    """Index project declarations once, before resolving any method bodies."""

    def __init__(self, trees: dict[str, Node], resolver_factory: Any):
        self.types: dict[str, TypeInfo] = {}
        self.by_node: dict[int, str] = {}
        declarations = []

        def register(node, prefix, file):
            name = f"{prefix}.{node.name}" if prefix else node.name
            declarations.append((name, file, node))
            self.by_node[id(node)] = name
            # Only member types are globally nameable, not local/anonymous classes.
            for child in members(node):
                if isinstance(child, jt.TypeDeclaration):
                    register(child, name, file)

        for file, tree in trees.items():
            package = tree.package.name if tree.package else ""
            for declaration in tree.types:
                register(declaration, package, file)
        known = {name for name, _, _ in declarations}
        for name, file, node in declarations:
            resolver = resolver_factory(trees[file], known)
            type_parameters = {p.name for p in getattr(node, "type_parameters", None) or []}
            for parent_name, _, parent_node in declarations:
                if name.startswith(parent_name + "."):
                    type_parameters.update(p.name for p in getattr(parent_node, "type_parameters", None) or [])
            resolver = replace(resolver, enclosing_type=name, type_parameters=frozenset(type_parameters))
            self.types[name] = TypeInfo(name, file, node, resolver)
        for info in self.types.values():
            parent_nodes = getattr(info.node, "extends", None)
            parent_nodes = parent_nodes if isinstance(parent_nodes, list) else [parent_nodes]
            parent_nodes += list(getattr(info.node, "implements", None) or [])
            info.parents = [resolved for p in parent_nodes
                            if (resolved := info.resolver.resolve(declared_name(p)))]
            for f in getattr(info.node, "fields", []) or []:
                for d in f.declarators:
                    info.fields[d.name] = None if d.dimensions else info.resolver.resolve(declared_name(f.type))
            for m in getattr(info.node, "methods", []) or []:
                method_resolver = replace(info.resolver, type_parameters=info.resolver.type_parameters | {
                    p.name for p in getattr(m, "type_parameters", None) or []
                })
                key = (m.name, len(m.parameters))
                info.methods.setdefault(key, []).append((
                    method_resolver.resolve(declared_name(m.return_type)),
                    bool(m.parameters and m.parameters[-1].varargs),
                    "static" in (m.modifiers or set()),
                ))

    def field_type(self, owner, member, visited=frozenset()):
        if owner not in self.types or owner in visited:
            return False, None
        info = self.types[owner]
        if member in info.fields:
            return True, info.fields[member]
        found = [self.field_type(p, member, visited | {owner}) for p in info.parents]
        values = [value for exists, value in found if exists]
        return (bool(values), values[0] if len(values) == 1 else None)

    def method(self, owner, member, arity, static_only=False, visited=frozenset()):
        if owner not in self.types or owner in visited:
            return None, None
        info = self.types[owner]
        candidates = [entry for (name, count), entries in info.methods.items()
                      if name == member for entry in entries
                      if (count == arity or (entry[1] and arity >= count - 1))
                      and (not static_only or entry[2])]
        if candidates:
            returns = {entry[0] for entry in candidates}
            return owner, next(iter(returns)) if len(returns) == 1 else None
        inherited = {self.method(p, member, arity, static_only, visited | {owner}) for p in info.parents}
        inherited.discard((None, None))
        return next(iter(inherited)) if len(inherited) == 1 else (None, None)


class ScopedCalls:
    def __init__(self, index: ProjectTypes, info: TypeInfo, collector):
        self.index, self.info, self.collector = index, info, collector
        self.scopes: list[dict[str, str | None]] = [{}]

    def unresolved(self, node, reason):
        pos = getattr(node, "position", None)
        self.collector.unresolved.append({
            "source_name": self.info.file, "enclosing_type": self.info.name,
            "member": str(getattr(node, "member", "")),
            "qualifier": str(getattr(node, "qualifier", "") or ""),
            "line": pos.line if pos else None, "reason": reason,
        })

    def symbol(self, name):
        for scope in reversed(self.scopes):
            if name in scope:
                return True, scope[name]  # unresolved locals still shadow fields/types
        return self.index.field_type(self.info.name, name)

    def receiver(self, qualifier):
        parts = qualifier.split(".")
        if parts[0] == "this":
            owner, start, is_type = self.info.name, 1, False
        else:
            exists, owner = self.symbol(parts[0])
            start, is_type = 1, False
            if not exists:
                owner = None
                # Resolve a real type prefix; subsequent segments must be fields.
                for end in range(len(parts), 0, -1):
                    candidate = self.info.resolver.resolve(".".join(parts[:end]))
                    if candidate:
                        owner, start, is_type = candidate, end, True
                        break
        for field_name in parts[start:]:
            _, owner = self.index.field_type(owner, field_name)
            is_type = False
        return owner, is_type

    def call(self, node, receiver=None, explicit=False):
        for arg in node.arguments or []:
            self.visit(arg)
        static_only = False
        if explicit:
            owner = receiver
        elif isinstance(node, jt.SuperMethodInvocation):
            parents = self.info.parents
            owner = parents[0] if len(parents) == 1 and not node.qualifier else None
        elif node.qualifier:
            owner, static_only = self.receiver(node.qualifier)
        else:
            owner = self.info.name
        target, returns = self.index.method(owner, node.member, len(node.arguments or []), static_only)
        if not explicit and not node.qualifier and not isinstance(node, jt.SuperMethodInvocation) and target is None:
            imports = self.info.resolver.static_imports
            explicit_imports = [(path, wildcard) for path, wildcard in imports
                                if not wildcard and path.rpartition(".")[2] == node.member]
            imports = explicit_imports or [(path, wildcard) for path, wildcard in imports if wildcard]
            candidates = set()
            for imported, wildcard in imports:
                type_name, _, member = imported.rpartition(".")
                if wildcard:
                    type_name, member = imported, node.member
                if member == node.member:
                    match = self.index.method(type_name, node.member, len(node.arguments or []), True)
                    if match[0]:
                        candidates.add(match)
            if len(candidates) == 1 and len(imports) == 1:
                target, returns = candidates.pop()
        if target:
            self.collector.add(self.info.file, self.index.types[target].file, "METHOD_CALL")
        else:
            self.unresolved(node, "receiver_or_method_unresolved_or_ambiguous")
        return returns

    def selectors(self, node, owner):
        for selector in getattr(node, "selectors", None) or []:
            if isinstance(selector, (jt.MethodInvocation, jt.SuperMethodInvocation)):
                owner = self.call(selector, owner, explicit=True)
            elif isinstance(selector, jt.MemberReference):
                _, owner = self.index.field_type(owner, selector.member)
            else:
                self.visit(selector)
                owner = None
        return owner

    def block(self, body):
        self.scopes.append({})
        self.visit(body)
        self.scopes.pop()

    def declare(self, node):
        type_name = self.info.resolver.resolve(declared_name(node.type))
        for d in node.declarators:
            self.scopes[-1][d.name] = None if d.dimensions else type_name
            self.visit(d.initializer)

    def visit(self, node):
        if isinstance(node, (list, tuple)):
            for child in node:
                self.visit(child)
            return None
        if not isinstance(node, Node):
            return None
        if isinstance(node, jt.TypeDeclaration):
            # Member types are visited independently; local classes lack a stable
            # project type identity and must not inherit the outer method receiver.
            if id(node) not in self.index.by_node:
                self.unresolved(node, "local_class_body_not_resolved")
            return None
        if isinstance(node, jt.BlockStatement):
            self.block(node.statements)
        elif isinstance(node, (jt.LocalVariableDeclaration, jt.VariableDeclaration)):
            self.declare(node)
        elif isinstance(node, (jt.MethodInvocation, jt.SuperMethodInvocation)):
            return self.selectors(node, self.call(node))
        elif isinstance(node, jt.This):
            owner = self.info.name if not node.qualifier else self.info.resolver.resolve(node.qualifier)
            return self.selectors(node, owner)
        elif isinstance(node, jt.MemberReference):
            if node.qualifier:
                owner, _ = self.receiver(node.qualifier)
                _, owner = self.index.field_type(owner, node.member)
            else:
                _, owner = self.symbol(node.member)
            return self.selectors(node, owner)
        elif isinstance(node, jt.ClassCreator):
            for arg in node.arguments or []:
                self.visit(arg)
            owner = self.info.resolver.resolve(declared_name(node.type))
            if node.body:
                self.unresolved(node, "anonymous_class_body_not_resolved")
            return self.selectors(node, owner)
        elif isinstance(node, jt.Cast):
            self.visit(node.expression)
            return self.info.resolver.resolve(declared_name(node.type))
        elif isinstance(node, jt.ForStatement):
            self.scopes.append({})
            control = node.control
            if isinstance(control, jt.EnhancedForControl):
                self.visit(control.iterable)
                self.declare(control.var)
            else:
                self.visit(control.init)
                self.visit(control.condition)
            self.block(node.body)
            if isinstance(control, jt.ForControl):
                self.visit(control.update)
            self.scopes.pop()
        elif isinstance(node, jt.TryStatement):
            self.scopes.append({})
            for resource in node.resources or []:
                self.scopes[-1][resource.name] = self.info.resolver.resolve(declared_name(resource.type))
                self.visit(resource.value)
            self.block(node.block)
            self.scopes.pop()
            for catch in node.catches or []:
                self.scopes.append({catch.parameter.name: None})
                if len(catch.parameter.types) == 1:
                    self.scopes[-1][catch.parameter.name] = self.info.resolver.resolve(catch.parameter.types[0])
                self.block(catch.block)
                self.scopes.pop()
            self.block(node.finally_block)
        elif isinstance(node, jt.LambdaExpression):
            self.scopes.append({})
            for parameter in node.parameters:
                name = getattr(parameter, "name", None) or getattr(parameter, "member", None)
                if name:
                    self.scopes[-1][name] = self.info.resolver.resolve(declared_name(getattr(parameter, "type", None)))
            self.block(node.body)
            self.scopes.pop()
        elif isinstance(node, jt.SwitchStatement):
            self.visit(node.expression)
            self.scopes.append({})
            self.visit(node.cases)
            self.scopes.pop()
        elif isinstance(node, jt.SynchronizedStatement):
            self.visit(node.lock)
            self.block(node.block)
        elif isinstance(node, jt.InnerClassCreator):
            for arg in node.arguments or []:
                self.visit(arg)
            self.unresolved(node, "qualified_inner_creation_not_resolved")
            return self.selectors(node, None)
        else:
            for attr in node.attrs:
                if attr == "selectors":
                    self.selectors(node, None)
                else:
                    self.visit(getattr(node, attr))
        return None

    def run(self):
        if isinstance(self.info.node, jt.EnumDeclaration):
            for constant in self.info.node.body.constants:
                self.block(constant.arguments)
                if constant.body:
                    self.unresolved(constant, "enum_constant_body_not_resolved")
        for field_node in getattr(self.info.node, "fields", []) or []:
            for d in field_node.declarators:
                self.block(d.initializer)
        for member in members(self.info.node):
            if isinstance(member, list):
                self.block(member)
        methods = executables(self.info.node)
        original_resolver = self.info.resolver
        for method in methods:
            self.info.resolver = replace(original_resolver, type_parameters=original_resolver.type_parameters | {
                p.name for p in getattr(method, "type_parameters", None) or []
            })
            self.scopes = [{}]
            for p in method.parameters:
                self.scopes[0][p.name] = None if p.varargs else self.info.resolver.resolve(declared_name(p.type))
            self.block(method.body)
        self.info.resolver = original_resolver
