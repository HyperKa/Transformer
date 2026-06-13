package org.example.parser.pipeline;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.ObjectCreationExpr;
import com.github.javaparser.ast.expr.StringLiteralExpr;
import com.github.javaparser.ast.stmt.CatchClause;
import org.example.parser.model.CodeCandidate;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.UUID;

public class Stage1AstExtractor {

    private static final int MIN_METHOD_LINES = 3;
    private static final int MAX_CONTEXT_LENGTH = 6000; // Ограничение длины строки контекста

    /**
     * Главный метод парсинга файла. Возвращает список кандидатов-методов с их контекстом.
     */
    public static List<CodeCandidate> extractCandidates(String fileContent, String sourceName, String filePath) {
        List<CodeCandidate> candidates = new ArrayList<>();

        try {
            // постройка AST (Абстрактное Синтаксическое Дерево, логика в записях)
            CompilationUnit cu = StaticJavaParser.parse(fileContent);

            // скан всех методов
            cu.findAll(MethodDeclaration.class).forEach(method -> {
                if (isMethodWorthAnalyzing(method)) {
                    String context = buildSmartContext(cu, method);

                    // если метод не больше 800 токенов - гуд
                    if (context.length() <= MAX_CONTEXT_LENGTH) {
                        CodeCandidate candidate = new CodeCandidate();
                        candidate.setId(UUID.randomUUID().toString());
                        candidate.setSource(sourceName);
                        candidate.setFilePath(filePath);
                        candidate.setVulnerableMethod(method.getNameAsString());
                        candidate.setFullContext(context);
                        candidate.setCodeLength(context.length());

                        // -1 означает "кандидат готов к анализу (неразмечен)"
                        candidate.setFinalClass(-1);

                        candidates.add(candidate);
                    }
                }
            });
        } catch (Exception e) {
            // Игнор синтаксически сломанных файлов или файлов старых версий Java
            // пока проблема в том, что ошибка разрыва соединения по VPN тоже попадает сюда
        }

        return candidates;
    }

    /**
     * Логика фильтрации: решение, отправлять ли метод в LLM.
     */
    private static boolean isMethodWorthAnalyzing(MethodDeclaration method) {
        // Отброс методов без тела (интерфейсы, абстрактные)
        // В отличие от фильтра на этапе загрузки, тут отсев на уровне методов.
        // Содержаться могут пустые методы внутри классов, они не имеют смысла (конструкторы)
        if (!method.getBody().isPresent()) return false;

        // 2. АБСОЛЮТНЫЙ ПРИОРИТЕТ: Жадный поиск маркеров (AST).
        // Если внутри есть БД, сеть, файлы, XML или пустые catch - берем 100%, даже если метод в 1 строку.
        if (hasAstSecurityMarkers(method)) {
            return true;
        }

        // этот этап для извлечения имен потенциально безопасных методов
        String name = method.getNameAsString();

        // Отброс DTO геттеров/сеттеров
        if (name.startsWith("get") || name.startsWith("set") || name.startsWith("is")) return false;

        // Отброс стандартных методов Object
        if (name.equals("equals") || name.equals("hashCode") || name.equals("toString")) return false;

        // Отброс коротких методов (тут остаются только вычисления, делегаты, статик-методы), не имеющие отношения к ресурсам
        int lines = method.getRange().map(r -> r.end.line - r.begin.line).orElse(0);
        return lines >= MIN_METHOD_LINES;
    }

    /**
     * Поиск маркеров безопасности через обход узлов дерева (AST).
     */
    private static boolean hasAstSecurityMarkers(MethodDeclaration method) {

        // поиск хотя бы 1 метода бд
        boolean hasDangerousCall = method.findAll(MethodCallExpr.class).stream()
                .map(call -> call.getNameAsString().toLowerCase())
                .anyMatch(name ->
                        // СТРОГОЕ совпадение (защита от мусора в Elasticsearch/Lucene)
                        name.equals("executequery") || name.equals("executeupdate") ||
                                name.equals("preparestatement") || name.equals("createstatement") ||
                                name.equals("createquery") || name.equals("createnativequery") ||
                                name.equals("exec") || name.equals("readobject") ||
                                name.equals("getunsafe") ||

                                // ШИРОКОЕ совпадение для кастомных/редких уязвимостей
                                name.contains("sql") ||          // Ловит: runSql, executeSql
                                name.contains("jdbc") ||         // Ловит: getJdbcTemplate
                                name.contains("xpath") ||        // Ловит: evaluateXPath (Класс 6)
                                name.contains("deserialize")     // Ловит: deserializeData (Класс 7)
                );
        if (hasDangerousCall) return true;

        // методы коннекта, билдеров
        boolean hasDangerousObjects = method.findAll(ObjectCreationExpr.class).stream()
                .map(obj -> obj.getType().getNameAsString().toLowerCase())
                .anyMatch(type ->
                        // Потоки ввода-вывода (Класс 1)
                        type.contains("inputstream") || type.contains("outputstream") ||
                                type.contains("reader") || type.contains("writer") ||
                                type.contains("connection") || type.equals("socket") ||
                                type.equals("serversocket") ||

                                // XML парсеры (Класс 3)
                                type.contains("saxparser") || type.contains("documentbuilder") ||
                                type.contains("xmlinputfactory") ||

                                // RCE (Класс 5)
                                type.equals("processbuilder")
                );
        if (hasDangerousObjects) return true;

        // поиск кастомного SQL-кода
        boolean hasSqlString = method.findAll(StringLiteralExpr.class).stream()
                .map(str -> str.getValue().toLowerCase())
                .anyMatch(val ->
                        (val.contains("select ") && val.contains(" from ")) ||
                                (val.contains("insert into ")) ||
                                (val.contains("update ") && val.contains(" set ")) ||
                                (val.contains("delete from "))
                );
        if (hasSqlString) return true;

        // пустые catch
        boolean hasEmptyCatch = method.findAll(CatchClause.class).stream()
                .anyMatch(catchClause -> catchClause.getBody().isEmpty() ||
                        catchClause.getBody().getStatements().isEmpty());
        if (hasEmptyCatch) return true;

        // поиск аннотаций
        boolean hasWebAnnotations = method.getAnnotations().stream()
                .map(a -> a.getNameAsString().toLowerCase())
                .anyMatch(name ->
                        name.contains("mapping") || // @GetMapping, @PostMapping (Spring)
                                name.equals("path")         // @Path (JAX-RS)
                );
        if (hasWebAnnotations) return true;

        return false;
    }

    /**
     * Склеивает контекст: Пакет + Импорты + Сигнатура + Поля + Метод
     */
    private static String buildSmartContext(CompilationUnit cu, MethodDeclaration method) {
        StringBuilder sb = new StringBuilder();
        String methodBody = method.toString();

        // 1. Пакет
        cu.getPackageDeclaration().ifPresent(p -> sb.append(p.toString()).append("\n"));

        // 2. Импорты: только те, что используются в методе
        cu.getImports().stream()
                .filter(im -> {
                    String className = im.getNameAsString();
                    String simpleName = className.substring(className.lastIndexOf(".") + 1);
                    return methodBody.contains(simpleName);
                })
                .forEach(i -> sb.append(i.toString()));
        sb.append("\n");

        Optional<ClassOrInterfaceDeclaration> classOpt = method.findAncestor(ClassOrInterfaceDeclaration.class);
        if (classOpt.isPresent()) {
            ClassOrInterfaceDeclaration clazz = classOpt.get();
            sb.append("class ").append(clazz.getNameAsString()).append(" {\n");

            // 3. Поля: Только те поля класса, которые используются в методе
            for (FieldDeclaration field : clazz.getFields()) {
                boolean isUsed = field.getVariables().stream()
                        .anyMatch(v -> methodBody.contains(v.getNameAsString()));
                if (isUsed) {
                    sb.append("    ").append(field.toString());
                }
            }
            sb.append("\n");
        }

        // 4. Собственно остается само тело метода
        sb.append(methodBody).append("\n");

        if (classOpt.isPresent()) sb.append("}");

        return sb.toString().trim();
    }
}