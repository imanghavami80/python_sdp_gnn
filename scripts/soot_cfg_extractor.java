import soot.*;
import soot.jimple.*;
import soot.options.Options;
import soot.toolkits.graph.ExceptionalUnitGraph;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class soot_cfg_extractor {
    private static final Set<String> EDGE_TYPES = new HashSet<>(Arrays.asList(
            "CFG_NEXT", "CFG_TRUE", "CFG_FALSE", "CFG_RETURN", "CFG_EXCEPTION"
    ));

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\").replace("\t", " ").replace("\n", " ").replace("\r", " ");
    }

    private static String nodeType(Unit u) {
        if (u instanceof IfStmt) return "CONDITION";
        if (u instanceof TableSwitchStmt || u instanceof LookupSwitchStmt) return "SWITCH";
        if (u instanceof ReturnStmt || u instanceof ReturnVoidStmt) return "RETURN";
        if (u instanceof ThrowStmt) return "THROW";
        return "STATEMENT";
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: soot_cfg_extractor <classes_dir> <class_list_txt> <source_root> <output_tsv>");
            System.exit(2);
        }

        String classesDir = args[0];
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
        Options.v().set_process_dir(Collections.singletonList(classesDir));
        String cp = classesDir + File.pathSeparator + sourceRoot + File.pathSeparator + System.getProperty("java.class.path");
        Options.v().set_soot_classpath(cp);

        Scene.v().loadNecessaryClasses();

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
                        w.write("NODE\t" + className + "\t" + methodId + "\t" + entry + "\tENTRY\t\t1\t__ENTRY__\n");
                        w.write("NODE\t" + className + "\t" + methodId + "\t" + exit + "\tEXIT\t\t1\t__EXIT__\n");

                        for (Unit u : units) {
                            int line = u.getJavaSourceStartLineNumber();
                            String lineStr = line > 0 ? Integer.toString(line) : "";
                            w.write("NODE\t" + className + "\t" + methodId + "\t" + uid.get(u) + "\t" + nodeType(u) + "\t" + lineStr + "\t0\t" + esc(u.toString()) + "\n");
                        }

                        for (Unit h : g.getHeads()) {
                            Integer hid = uid.get(h);
                            if (hid != null) {
                                w.write("EDGE\t" + className + "\t" + methodId + "\t" + entry + "\t" + hid + "\tCFG_NEXT\n");
                            }
                        }

                        for (Unit u : units) {
                            int src = uid.get(u);
                            boolean hasAnyOut = false;

                            if (u instanceof ReturnStmt || u instanceof ReturnVoidStmt) {
                                hasAnyOut = true;
                                w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + exit + "\tCFG_RETURN\n");
                            }

                            List<Unit> succs = g.getUnexceptionalSuccsOf(u);
                            if (u instanceof IfStmt) {
                                Unit t = ((IfStmt) u).getTarget();
                                for (Unit s : succs) {
                                    Integer tid = uid.get(s);
                                    if (tid == null) continue;
                                    hasAnyOut = true;
                                    String et = s.equals(t) ? "CFG_TRUE" : "CFG_FALSE";
                                    w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + tid + "\t" + et + "\n");
                                }
                            } else {
                                for (Unit s : succs) {
                                    Integer tid = uid.get(s);
                                    if (tid == null) continue;
                                    hasAnyOut = true;
                                    w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + tid + "\tCFG_NEXT\n");
                                }
                            }

                            List<Unit> exSuccs = g.getExceptionalSuccsOf(u);
                            for (Unit s : exSuccs) {
                                Integer tid = uid.get(s);
                                if (tid == null) continue;
                                hasAnyOut = true;
                                w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + tid + "\tCFG_EXCEPTION\n");
                            }

                            if (!hasAnyOut && !(u instanceof ThrowStmt)) {
                                w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + exit + "\tCFG_NEXT\n");
                            }
                            if (u instanceof ThrowStmt) {
                                w.write("EDGE\t" + className + "\t" + methodId + "\t" + src + "\t" + exit + "\tCFG_EXCEPTION\n");
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
