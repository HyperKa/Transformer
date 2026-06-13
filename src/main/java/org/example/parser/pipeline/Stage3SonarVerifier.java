package org.example.parser.pipeline;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.*;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.TryStmt;
import org.example.parser.model.CodeCandidate;
import org.example.parser.model.SonarVerdict;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

public class Stage3SonarVerifier {

    public SonarVerdict analyze(CodeCandidate candidate) {
        try {
            // AST дерево с кодом тела примера
            CompilationUnit cu = StaticJavaParser.parse(candidate.getFullContext());
            String fullCode = candidate.getFullContext().toLowerCase();

            Set<String> taintedVars = new HashSet<>();
            cu.findAll(VariableDeclarator.class).forEach(v -> {
                v.getInitializer().ifPresent(init -> {
                    if (init.isBinaryExpr()) taintedVars.add(v.getNameAsString());
                });
            });

            // RESOURCE_LEAK (Класс 1)
            boolean hasS2095 = cu.findAll(ObjectCreationExpr.class).stream()
                    .anyMatch(oc -> {
                        String type = oc.getType().getNameAsString().toLowerCase();
                        boolean isRes = type.contains("stream") || type.contains("socket") ||
                                type.contains("connection") || type.contains("writer");
                        boolean isWrapped = oc.findAncestor(TryStmt.class)
                                .map(t -> !t.getResources().isEmpty()).orElse(false);
                        return isRes && !isWrapped;
                    });
            if (hasS2095) return new SonarVerdict(1, 1.0, "java:S2095 (Leak)");

            //  BAD_LOGIC_AND_EXCEPTION (Класс 2)
            boolean hasS112 = cu.findAll(CatchClause.class).stream()
                    .anyMatch(cc -> cc.getBody().getStatements().isEmpty() ||
                            cc.getParameter().getType().asString().equals("Exception"));
            if (hasS112 || fullCode.contains("catch (throwable") || fullCode.contains("integer.max_value"))
                return new SonarVerdict(2, 1.0, "java:S112 (Bad Logic)");

            // XXE (Класс 3)
            boolean hasS2755 = fullCode.contains("documentbuilderfactory") ||
                    fullCode.contains("xmlinputfactory") ||
                    fullCode.contains("saxparserfactory");
            if (hasS2755 && !fullCode.contains("disallow-doctype-decl"))
                return new SonarVerdict(3, 1.0, "java:S2755 (XXE)");

            // NPE (Класс 4)
            // Ловит случай: if (obj == null) { ... } obj.method();
            boolean hasNpeLogic = cu.findAll(MethodCallExpr.class).stream()
                    .anyMatch(mc -> mc.getScope().isPresent() &&
                            fullCode.contains(mc.getScope().get().toString().toLowerCase() + " == null"));
            if (hasNpeLogic || fullCode.contains("return null"))
                return new SonarVerdict(4, 1.0, "java:S2259 (NPE)");

            // RCE (Класс 5)
            boolean hasS2076 = cu.findAll(MethodCallExpr.class).stream()
                    .anyMatch(mc -> {
                        String name = mc.getNameAsString();
                        boolean isExec = name.equals("exec") || name.equals("start") || name.equals("loadLibrary");
                        boolean hasVarArg = mc.getArguments().stream().anyMatch(arg -> !arg.isStringLiteralExpr());
                        return isExec && hasVarArg;
                    });
            if (hasS2076 || fullCode.contains("processbuilder"))
                return new SonarVerdict(5, 1.0, "java:S2076 (RCE)");

            // SQL/LDAP INJECTION (Класс 6)
            boolean hasS3649 = cu.findAll(MethodCallExpr.class).stream()
                    .anyMatch(mc -> {
                        String n = mc.getNameAsString().toLowerCase();
                        boolean isSink = n.contains("execute") || n.contains("query") || n.contains("search");
                        // либо аргумент - конкатенация, либо это "грязная" переменная из списка
                        boolean hasTaint = mc.getArguments().stream().anyMatch(arg ->
                                arg.isBinaryExpr() || (arg.isNameExpr() && taintedVars.contains(arg.toString())));
                        return isSink && hasTaint;
                    });
            if (hasS3649) return new SonarVerdict(6, 1.0, "java:S3649 (Injection)");

            // UNSAFE_API (Класс 7)
            boolean hasClass7 = fullCode.contains("readobject") ||
                    fullCode.contains("sun.misc.unsafe") ||
                    fullCode.contains("cryptography.getinstance(\"des\")") || // Слабое шифрование
                    fullCode.contains("cookie.setsecure(false)");
            if (hasClass7) return new SonarVerdict(7, 1.0, "java:S502 (Unsafe API)");

        } catch (Exception e) {
            return new SonarVerdict(0, 0.0, "PARSE_ERROR");
        }

        return new SonarVerdict(0, 0.0, "NONE");
    }
}