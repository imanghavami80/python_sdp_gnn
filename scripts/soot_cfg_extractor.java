import soot.*;
import soot.jimple.*;
import soot.options.Options;
import soot.toolkits.graph.ExceptionalUnitGraph;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.function.Predicate;

public class soot_cfg_extractor {
    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r", " ");
    }

    private static String nodeType(Unit u) {
        if (u instanceof IdentityStmt && ((IdentityStmt) u).getRightOp() instanceof CaughtExceptionRef) return "CATCH";
        if (u instanceof IfStmt) return "CONDITION";
        if (u instanceof TableSwitchStmt || u instanceof LookupSwitchStmt) return "SWITCH";
        if (u instanceof ReturnStmt || u instanceof ReturnVoidStmt) return "RETURN";
        if (u instanceof ThrowStmt) return "THROW";
        return "STATEMENT";
    }

    private static String stmtKind(Unit u) {
        if (u instanceof IfStmt) return "IF";
        if (u instanceof GotoStmt) return "GOTO";
        if (u instanceof TableSwitchStmt || u instanceof LookupSwitchStmt) return "SWITCH";
        if (u instanceof ReturnStmt) return "RETURN_VALUE";
        if (u instanceof ReturnVoidStmt) return "RETURN_VOID";
        if (u instanceof ThrowStmt) return "THROW";
        if (u instanceof EnterMonitorStmt || u instanceof ExitMonitorStmt) return "MONITOR";
        if (u instanceof NopStmt) return "NOP";
        if (u instanceof IdentityStmt) return "IDENTITY";
        if (u instanceof AssignStmt) return "ASSIGN";
        if (u instanceof InvokeStmt) return "INVOKE";
        return "OTHER";
    }

    private static InvokeExpr invokeExpr(Unit u) {
        if (u instanceof Stmt) {
            Stmt s = (Stmt) u;
            if (s.containsInvokeExpr()) return s.getInvokeExpr();
        }
        return null;
    }

    private static String invokeKind(Unit u) {
        InvokeExpr expr = invokeExpr(u);
        if (expr == null) return "NO_INVOKE";
        if (expr instanceof StaticInvokeExpr) return "STATIC_INVOKE";
        if (expr instanceof VirtualInvokeExpr) return "VIRTUAL_INVOKE";
        if (expr instanceof InterfaceInvokeExpr) return "INTERFACE_INVOKE";
        if (expr instanceof SpecialInvokeExpr) return "SPECIAL_INVOKE";
        if (expr instanceof DynamicInvokeExpr) return "DYNAMIC_INVOKE";
        return "UNKNOWN_INVOKE";
    }

    private static boolean valueMatches(Value v, Predicate<Value> pred, Set<Value> seen) {
        if (v == null) return false;
        if (seen.contains(v)) return false;
        seen.add(v);
        if (pred.test(v)) return true;
        for (ValueBox box : v.getUseBoxes()) {
            if (valueMatches(box.getValue(), pred, seen)) return true;
        }
        return false;
    }

    private static boolean unitMatches(Unit u, Predicate<Value> pred) {
        for (ValueBox box : u.getUseBoxes()) {
            if (valueMatches(box.getValue(), pred, new HashSet<Value>())) return true;
        }
        for (ValueBox box : u.getDefBoxes()) {
            if (valueMatches(box.getValue(), pred, new HashSet<Value>())) return true;
        }
        return false;
    }

    private static boolean unitUseMatches(Unit u, Predicate<Value> pred) {
        for (ValueBox box : u.getUseBoxes()) {
            if (valueMatches(box.getValue(), pred, new HashSet<Value>())) return true;
        }
        return false;
    }

    private static boolean isArithmeticValue(Value v) {
        return v instanceof AddExpr || v instanceof SubExpr || v instanceof MulExpr || v instanceof DivExpr || v instanceof RemExpr
                || v instanceof ShlExpr || v instanceof ShrExpr || v instanceof UshrExpr
                || v instanceof AndExpr || v instanceof OrExpr || v instanceof XorExpr || v instanceof NegExpr;
    }

    private static boolean isComparisonValue(Value v) {
        return v instanceof ConditionExpr || v instanceof CmpExpr || v instanceof CmpgExpr || v instanceof CmplExpr;
    }

    private static boolean hasFieldWrite(Unit u) {
        return u instanceof AssignStmt && ((AssignStmt) u).getLeftOp() instanceof FieldRef;
    }

    private static boolean hasArrayWrite(Unit u) {
        return u instanceof AssignStmt && ((AssignStmt) u).getLeftOp() instanceof ArrayRef;
    }

    private static String bit(boolean value) {
        return value ? "1" : "0";
    }

    private static String instructionFlags(Unit u) {
        boolean hasMethodCall = invokeExpr(u) != null;
        boolean hasFieldRead = unitUseMatches(u, v -> v instanceof FieldRef);
        boolean hasFieldWrite = hasFieldWrite(u);
        boolean hasArrayRead = unitUseMatches(u, v -> v instanceof ArrayRef);
        boolean hasArrayWrite = hasArrayWrite(u);
        boolean hasNewObject = unitMatches(u, v -> v instanceof NewExpr);
        boolean hasNewArray = unitMatches(u, v -> v instanceof NewArrayExpr || v instanceof NewMultiArrayExpr);
        boolean hasCast = unitMatches(u, v -> v instanceof CastExpr);
        boolean hasArithmeticOp = unitMatches(u, soot_cfg_extractor::isArithmeticValue);
        boolean hasComparisonOp = unitMatches(u, soot_cfg_extractor::isComparisonValue);
        boolean hasNullConstant = unitMatches(u, v -> v instanceof NullConstant);
        boolean hasStringConstant = unitMatches(u, v -> v instanceof StringConstant);
        boolean hasNumericConstant = unitMatches(u, v -> v instanceof NumericConstant);

        return String.join("\t", Arrays.asList(
                bit(hasMethodCall),
                bit(hasFieldRead),
                bit(hasFieldWrite),
                bit(hasArrayRead),
                bit(hasArrayWrite),
                bit(hasNewObject),
                bit(hasNewArray),
                bit(hasCast),
                bit(hasArithmeticOp),
                bit(hasComparisonOp),
                bit(hasNullConstant),
                bit(hasStringConstant),
                bit(hasNumericConstant)
        ));
    }

    private static void writeEdge(BufferedWriter w, String className, String methodId, int src, int dst, String edgeType) throws IOException {
        w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + dst + "\t" + edgeType + "\n");
    }

    private static void writeUnitEdge(
            BufferedWriter w,
            String className,
            String methodId,
            Unit srcUnit,
            Unit dstUnit,
            String edgeType,
            Map<Unit, Integer> uid
    ) throws IOException {
        Integer src = uid.get(srcUnit);
        Integer dst = uid.get(dstUnit);
        if (src == null || dst == null) return;
        writeEdge(w, className, methodId, src, dst, edgeType);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: soot_cfg_extractor <bytecode_classpath> <class_list_txt> <source_root> <output_tsv>");
            System.exit(2);
        }

        String bytecodeClasspath = args[0];
        Path classListPath = Paths.get(args[1]);
        String sourceRoot = args[2];
        Path outPath = Paths.get(args[3]);

        List<String> classes = Files.readAllLines(classListPath, StandardCharsets.UTF_8);

        G.reset();
        Options.v().set_prepend_classpath(true);
        Options.v().set_allow_phantom_refs(true);
        Options.v().set_whole_program(false);
        Options.v().set_output_format(Options.output_format_none);
        Options.v().set_keep_line_number(true);
        Options.v().set_src_prec(Options.src_prec_only_class);
        String cp = bytecodeClasspath + File.pathSeparator + sourceRoot + File.pathSeparator + System.getProperty("java.class.path");
        Options.v().set_soot_classpath(cp);
        Scene.v().loadBasicClasses();

        try (BufferedWriter w = Files.newBufferedWriter(outPath, StandardCharsets.UTF_8)) {
            for (String rawName : classes) {
                String className = rawName.trim();
                if (className.isEmpty()) continue;
                try {
                    SootClass sc = Scene.v().forceResolve(className, SootClass.BODIES);
                    if (sc == null) {
                        w.write("FAIL\t" + className + "\tclass_resolve_null\n");
                        continue;
                    }
                    sc.setApplicationClass();

                    int mIndex = 0;
                    String simple = className.contains(".") ? className.substring(className.lastIndexOf('.') + 1) : className;
                    for (SootMethod m : sc.getMethods()) {
                        if (!m.isConcrete()) {
                            mIndex++;
                            continue;
                        }

                        Body b;
                        try {
                            b = m.retrieveActiveBody();
                        } catch (Exception ex) {
                            w.write("MFAIL\t" + className + "\t" + esc(m.getSubSignature()) + "\t" + esc(ex.toString()) + "\n");
                            mIndex++;
                            continue;
                        }

                        boolean compilerProblemBody = false;
                        for (Unit u : b.getUnits()) {
                            if (u.toString().contains("Unresolved compilation problem")) {
                                compilerProblemBody = true;
                                break;
                            }
                        }
                        if (compilerProblemBody) {
                            w.write("MFAIL\t" + className + "\t" + esc(m.getSubSignature()) + "\tcompiler_problem_body\n");
                            mIndex++;
                            continue;
                        }

                        String methodId = simple + "." + m.getName() + "#" + mIndex + "[p" + m.getParameterCount() + "]";
                        mIndex++;

                        ExceptionalUnitGraph g = new ExceptionalUnitGraph(b);
                        List<Unit> units = new ArrayList<>(b.getUnits());
                        Map<Unit, Integer> uid = new HashMap<>();
                        int nextId = 2;
                        for (Unit u : units) {
                            uid.put(u, nextId++);
                        }
                        int entry = 0;
                        int exit = 1;

                        w.write("METHOD\t" + className + "\t" + methodId + "\t" + (m.isConstructor() ? "constructor" : "method") + "\t" + entry + "\t" + exit + "\n");
                        w.write("NODE\t" + className + "\t" + methodId + "\t" + entry + "\tENTRY\t\t1\tNO_STMT\tNO_INVOKE\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t__ENTRY__\n");
                        w.write("NODE\t" + className + "\t" + methodId + "\t" + exit + "\tEXIT\t\t1\tNO_STMT\tNO_INVOKE\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\t__EXIT__\n");

                        for (Unit u : units) {
                            int line = u.getJavaSourceStartLineNumber();
                            String lineStr = line > 0 ? Integer.toString(line) : "";
                            w.write(
                                    "NODE\t" + className + "\t" + methodId + "\t" + uid.get(u) + "\t" + nodeType(u) + "\t" + lineStr
                                            + "\t0\t" + stmtKind(u) + "\t" + invokeKind(u) + "\t" + instructionFlags(u) + "\t" + esc(u.toString()) + "\n"
                            );
                        }

                        for (Unit h : g.getHeads()) {
                            Integer hid = uid.get(h);
                            if (hid != null) {
                                writeEdge(w, className, methodId, entry, hid, "CFG_ENTRY");
                            }
                        }

                        for (Unit u : units) {
                            int src = uid.get(u);
                            boolean hasAnyOut = false;

                            if (u instanceof ReturnStmt || u instanceof ReturnVoidStmt) {
                                hasAnyOut = true;
                                writeEdge(w, className, methodId, src, exit, "CFG_RETURN");
                            }

                            List<Unit> succs = g.getUnexceptionalSuccsOf(u);
                            if (u instanceof IfStmt) {
                                Unit t = ((IfStmt) u).getTarget();
                                for (Unit s : succs) {
                                    if (!uid.containsKey(s)) continue;
                                    hasAnyOut = true;
                                    String et = s.equals(t) ? "CFG_BRANCH_TRUE" : "CFG_BRANCH_FALSE";
                                    writeUnitEdge(w, className, methodId, u, s, et, uid);
                                }
                            } else if (u instanceof TableSwitchStmt || u instanceof LookupSwitchStmt) {
                                Unit defaultTarget;
                                if (u instanceof TableSwitchStmt) {
                                    defaultTarget = ((TableSwitchStmt) u).getDefaultTarget();
                                } else {
                                    defaultTarget = ((LookupSwitchStmt) u).getDefaultTarget();
                                }
                                for (Unit s : succs) {
                                    if (!uid.containsKey(s)) continue;
                                    hasAnyOut = true;
                                    String et = s.equals(defaultTarget) ? "CFG_SWITCH_DEFAULT" : "CFG_SWITCH_CASE";
                                    writeUnitEdge(w, className, methodId, u, s, et, uid);
                                }
                            } else if (u instanceof GotoStmt) {
                                for (Unit s : succs) {
                                    if (!uid.containsKey(s)) continue;
                                    hasAnyOut = true;
                                    writeUnitEdge(w, className, methodId, u, s, "CFG_GOTO", uid);
                                }
                            } else {
                                for (Unit s : succs) {
                                    if (!uid.containsKey(s)) continue;
                                    hasAnyOut = true;
                                    writeUnitEdge(w, className, methodId, u, s, "CFG_FALLTHROUGH", uid);
                                }
                            }

                            List<Unit> exSuccs = g.getExceptionalSuccsOf(u);
                            for (Unit s : exSuccs) {
                                Integer tid = uid.get(s);
                                if (tid == null) continue;
                                hasAnyOut = true;
                                writeEdge(w, className, methodId, src, tid, "CFG_EXCEPTION_HANDLER");
                            }

                            if (!hasAnyOut && !(u instanceof ThrowStmt)) {
                                writeEdge(w, className, methodId, src, exit, "CFG_FALLTHROUGH");
                            }

                            boolean exceptionEscapes = false;
                            for (ExceptionalUnitGraph.ExceptionDest dest : g.getExceptionDests(u)) {
                                if (dest.getTrap() == null && !dest.getThrowables().isEmpty()) {
                                    exceptionEscapes = true;
                                    break;
                                }
                            }
                            if (exceptionEscapes) {
                                writeEdge(
                                        w, className, methodId, src, exit,
                                        u instanceof ThrowStmt ? "CFG_THROW" : "CFG_EXCEPTION_EXIT"
                                );
                            }
                        }
                    }
                } catch (Exception ex) {
                    w.write("FAIL\t" + className + "\t" + esc(ex.toString()) + "\n");
                }
            }
        }
    }
}
