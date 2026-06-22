package org.example.parser.source;

import org.example.parser.model.CodeCandidate;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.*;
import java.util.regex.*;
import java.util.stream.Collectors;

public class JulietLoader {

    private final Path archivePath;
    private final Path extractPath;
    private final Path outputPath;

    // Маппинг CWE → класс уязвимости (0-7)
    private static final Map<String, Integer> CWE_TO_CLASS = Map.ofEntries(
            // Класс 1: RESOURCE_LEAK (Утечки)
            Map.entry("CWE400_Resource_Exhaustion", 1),
            Map.entry("CWE404_Improper_Resource_Shutdown", 1),
            Map.entry("CWE772_Missing_Release_of_Resource", 1),
            Map.entry("CWE775_Missing_Release_of_File_Descriptor_or_Handle", 1),

            // Класс 2: BAD_LOGIC_AND_EXCEPTION (Ошибки логики и переполнения)
            Map.entry("CWE190_Integer_Overflow", 2),
            Map.entry("CWE191_Integer_Underflow", 2),
            Map.entry("CWE197_Numeric_Truncation_Error", 2),
            Map.entry("CWE248_Uncaught_Exception", 2),
            Map.entry("CWE369_Divide_by_Zero", 2),
            Map.entry("CWE390_Error_Without_Action", 2),
            Map.entry("CWE395_Catch_NullPointerException", 2),
            Map.entry("CWE396_Catch_Generic_Exception", 2),
            Map.entry("CWE397_Throw_Generic", 2),
            Map.entry("CWE561_Dead_Code", 0), // Мертвый код помечаем как SAFE (0)
            Map.entry("CWE570_Expression_Always_False", 2),
            Map.entry("CWE571_Expression_Always_True", 2),
            Map.entry("CWE681_Incorrect_Conversion_Between_Numeric_Types", 2),

            // Класс 4: NPE (Нулевые указатели - ИСПРАВЛЕН РЕГИСТР NULL)
            Map.entry("CWE476_NULL_Pointer_Dereference", 4),
            Map.entry("CWE690_NULL_Deref_From_Return", 4),

            // Класс 5: RCE (Командные инъекции)
            Map.entry("CWE78_OS_Command_Injection", 5),
            Map.entry("CWE114_Process_Control", 5),

            // Класс 6: INJECTION (SQL/LDAP/XPath)
            Map.entry("CWE89_SQL_Injection", 6),
            Map.entry("CWE90_LDAP_Injection", 6),
            Map.entry("CWE643_Xpath_Injection", 6),

            // Класс 7: UNSAFE_API (Криптография и рефлексия)
            Map.entry("CWE327_Use_Broken_Crypto", 7),
            Map.entry("CWE470_Unsafe_Reflection", 7),
            Map.entry("CWE759_Unsalted_One_Way_Hash", 7),
            Map.entry("CWE760_Predictable_Salt_One_Way_Hash", 7)
    );

    public JulietLoader(String archiveZipPath, String extractFolderPath, String outputFolderPath)
            throws IOException {
        this.archivePath = Paths.get(archiveZipPath);
        this.extractPath = Paths.get(extractFolderPath);
        this.outputPath = Paths.get(outputFolderPath);

        Files.createDirectories(extractPath);
        Files.createDirectories(outputPath);
    }

    public void extractIfNeeded() throws IOException {
        if (hasJavaFiles(extractPath)) {
            System.out.println("Архив уже распакован: " + extractPath);
            return;
        }

        System.out.println("Распаковка архива: " + archivePath);

        try (ZipInputStream zis = new ZipInputStream(Files.newInputStream(archivePath))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                Path entryPath = extractPath.resolve(entry.getName());

                if (entry.isDirectory()) {
                    Files.createDirectories(entryPath);
                } else {
                    Files.createDirectories(entryPath.getParent());
                    Files.copy(zis, entryPath, StandardCopyOption.REPLACE_EXISTING);
                }
                zis.closeEntry();
            }
        }

        System.out.println("Распаковка завершена: " + extractPath);
    }

    private boolean hasJavaFiles(Path dir) throws IOException {
        if (!Files.exists(dir)) return false;
        try (var stream = Files.walk(dir)) {
            return stream.anyMatch(p -> p.toString().endsWith(".java"));
        }
    }

    public List<Path> findAllJavaFiles() throws IOException {
        List<Path> javaFiles = new ArrayList<>();
        try (var stream = Files.walk(extractPath)) {
            stream.filter(p -> p.toString().endsWith(".java"))
                    .forEach(javaFiles::add);
        }
        return javaFiles;
    }

    private String extractCweFromPath(Path path) {
        String pathStr = path.toString();
        Pattern pattern = Pattern.compile("CWE\\d+_\\w+");
        Matcher matcher = pattern.matcher(pathStr);
        if (matcher.find()) {
            return matcher.group();
        }
        Pattern pattern2 = Pattern.compile("CWE\\d+");
        matcher = pattern2.matcher(pathStr);
        if (matcher.find()) {
            return matcher.group() + "_General";
        }
        return null;
    }

    /**
     * Извлекает сигнатуру класса (package, imports, class declaration)
     */
    private String extractClassContext(String content) {
        StringBuilder context = new StringBuilder();

        Pattern packagePattern = Pattern.compile("^package\\s+[\\w.]+;", Pattern.MULTILINE);
        Matcher packageMatcher = packagePattern.matcher(content);
        if (packageMatcher.find()) {
            context.append(packageMatcher.group()).append("\n");
        }

        Pattern importPattern = Pattern.compile("^import\\s+[\\w.]+;", Pattern.MULTILINE);
        Matcher importMatcher = importPattern.matcher(content);
        int importCount = 0;
        while (importMatcher.find() && importCount < 15) {
            context.append(importMatcher.group()).append("\n");
            importCount++;
        }

        Pattern classPattern = Pattern.compile(
                "(public|private|protected)?\\s*(abstract|final)?\\s*class\\s+\\w+\\s*(extends\\s+\\w+)?\\s*(implements\\s+[^{]+)?\\s*\\{",
                Pattern.DOTALL
        );
        Matcher classMatcher = classPattern.matcher(content);
        if (classMatcher.find()) {
            context.append(classMatcher.group());
        }

        return context.toString();
    }

    /**
     * Извлекает полный метод с телом (до 6000 символов)
     */
    private String extractFullMethod(String content, String methodName) {
        Pattern pattern = Pattern.compile(
                "(public|private|protected)\\s+\\w+\\s+" + Pattern.quote(methodName) + "\\s*\\([^)]*\\)\\s*(?:throws\\s+\\w+(?:,\\s*\\w+)*)?\\s*\\{",
                Pattern.DOTALL
        );

        Matcher matcher = pattern.matcher(content);
        if (!matcher.find()) return null;

        int start = matcher.start();
        int braceCount = 0;
        int end = start;

        for (int i = start; i < content.length(); i++) {
            char c = content.charAt(i);
            if (c == '{') braceCount++;
            if (c == '}') {
                braceCount--;
                if (braceCount == 0) {
                    end = i + 1;
                    break;
                }
            }
        }

        String fullMethod = content.substring(start, end);
        return fullMethod.length() > 6000 ? fullMethod.substring(0, 6000) : fullMethod;
    }

    /**
     * Находит ВСЕ методы в файле, которые начинаются с указанного префикса
     * Например: "bad" → bad()
     *          "good" → good(), goodG2B(), goodG2B1(), goodG2B2(), goodB2G(), goodB2G1() и т.д.
     */
    private List<String> findAllMethodsByPrefix(String content, String prefix) {
        List<String> methods = new ArrayList<>();

        // поиск всех методов с заданным префиксом
        Pattern methodPattern = Pattern.compile(
                "(public|private|protected)\\s+\\w+\\s+(" + Pattern.quote(prefix) + "\\w*)\\s*\\([^)]*\\)\\s*(?:throws\\s+\\w+(?:,\\s*\\w+)*)?\\s*\\{",
                Pattern.DOTALL
        );

        Matcher matcher = methodPattern.matcher(content);
        while (matcher.find()) {
            String methodName = matcher.group(2);
            int start = matcher.start();
            int braceCount = 0;
            int end = start;

            for (int i = start; i < content.length(); i++) {
                char c = content.charAt(i);
                if (c == '{') braceCount++;
                if (c == '}') {
                    braceCount--;
                    if (braceCount == 0) {
                        end = i + 1;
                        break;
                    }
                }
            }

            String fullMethod = content.substring(start, end);
            if (fullMethod.length() > 100) {
                methods.add(fullMethod.length() > 6000 ? fullMethod.substring(0, 6000) : fullMethod);
            }
        }

        return methods;
    }

    /**
     * Создание кандидата
     */
    private CodeCandidate createCandidate(String cweName, int finalClass,
                                          String methodType, String fullContext,
                                          String methodCode, String filePath) {
        CodeCandidate candidate = new CodeCandidate();
        candidate.setId(UUID.randomUUID().toString());
        candidate.setSource("JULIET");
        candidate.setCwe(cweName);
        candidate.setFinalClass(finalClass);
        candidate.setMethodType(methodType);
        candidate.setFullContext(fullContext);
        candidate.setVulnerableMethod(methodCode);
        candidate.setFilePath(filePath);
        candidate.setCodeLength(fullContext.length());
        candidate.setCreatedAt(java.time.LocalDateTime.now());
        return candidate;
    }

    /**
     * Главный метод загрузки всех данных из Juliet
     * Находит ВСЕ bad-методы и ВСЕ good-методы (good, goodG2B, goodG2B1, goodB2G, goodB2G1...)
     */
    public List<CodeCandidate> loadAll() throws IOException {
        List<CodeCandidate> candidates = new ArrayList<>();

        extractIfNeeded();

        List<Path> javaFiles = findAllJavaFiles();
        System.out.println("Найдено Java файлов: " + javaFiles.size());

        int processedCount = 0;
        int badMethodsFound = 0;
        int goodMethodsFound = 0;
        int skippedNoCwe = 0;

        for (Path file : javaFiles) {
            String cweName = extractCweFromPath(file);

            if (cweName == null) {
                skippedNoCwe++;
                continue;
            }

            Integer classId = CWE_TO_CLASS.get(cweName);
            if (classId == null) {
                continue;
            }

            String content;
            try {
                content = Files.readString(file);
            } catch (Exception e) {
                System.err.println("Ошибка чтения файла: " + file);
                continue;
            }

            // Извлечение контекста класса
            String classContext = extractClassContext(content);

            // 1. Находим ВСЕ bad-методы
            List<String> badMethods = findAllMethodsByPrefix(content, "bad");
            for (String badMethod : badMethods) {
                // ДОБАВЛЯЕМ "\n}" в конце, чтобы закрыть класс!
                String fullContext = classContext + "\n" + badMethod + "\n}";
                candidates.add(createCandidate(cweName, classId, "bad",
                        fullContext, badMethod, file.toString()));
                badMethodsFound++;
            }

            // 2. Находим ВСЕ good-методы
            List<String> goodMethods = findAllMethodsByPrefix(content, "good");
            for (String goodMethod : goodMethods) {
                // ДОБАВЛЯЕМ "\n}" в конце, чтобы закрыть класс!
                String fullContext = classContext + "\n" + goodMethod + "\n}";
                candidates.add(createCandidate(cweName, 0, "good",
                        fullContext, goodMethod, file.toString()));
                goodMethodsFound++;
            }

            processedCount++;
            if (processedCount % 1000 == 0) {
                System.out.printf("Обработано %d файлов, bad=%d, good=%d, пропущено(noCWE)=%d%n",
                        processedCount, badMethodsFound, goodMethodsFound, skippedNoCwe);
            }
        }

        System.out.println("\n=== ИТОГО ПО JULIET ===");
        System.out.println("Обработано файлов: " + processedCount);
        System.out.println("Пропущено (нет CWE в пути): " + skippedNoCwe);
        System.out.println("Найдено bad-методов (уязвимых): " + badMethodsFound);
        System.out.println("Найдено good-методов (безопасных): " + goodMethodsFound);
        System.out.println("Всего кандидатов: " + candidates.size());

        // Статистика по длине контекста
        if (!candidates.isEmpty()) {
            IntSummaryStatistics stats = candidates.stream()
                    .mapToInt(CodeCandidate::getCodeLength)
                    .summaryStatistics();
            System.out.println("\n=== СТАТИСТИКА ДЛИНЫ КОНТЕКСТА ===");
            System.out.println("Мин: " + stats.getMin() + " символов");
            System.out.println("Макс: " + stats.getMax() + " символов");
            System.out.println("Среднее: " + Math.round(stats.getAverage()) + " символов");

            // Статистика по типам методов (уникальные имена)
            System.out.println("\n=== УНИКАЛЬНЫЕ ТИПЫ МЕТОДОВ ===");
            Set<String> uniqueTypes = candidates.stream()
                    .map(CodeCandidate::getMethodType)
                    .collect(Collectors.toSet());
            uniqueTypes.forEach(type -> System.out.println("  - " + type));
        }

        return candidates;
    }

    public static void main(String[] args) {
        try {
            String archivePath = "datasets/raw/2017-10-01-juliet-test-suite-for-java-v1-3.zip";
            String extractPath = "datasets/extracted/juliet";
            String outputPath = "datasets/output";

            JulietLoader loader = new JulietLoader(archivePath, extractPath, outputPath);
            List<CodeCandidate> candidates = loader.loadAll();

            // Сохранение в JSON
            Path outputFile = Paths.get(outputPath, "juliet_candidates.json");
            try (PrintWriter writer = new PrintWriter(Files.newBufferedWriter(outputFile))) {
                writer.println("[");
                for (int i = 0; i < candidates.size(); i++) {
                    CodeCandidate c = candidates.get(i);
                    writer.println("  {");
                    writer.println("    \"id\": \"" + c.getId() + "\",");
                    writer.println("    \"source\": \"" + c.getSource() + "\",");
                    writer.println("    \"cwe\": \"" + c.getCwe() + "\",");
                    writer.println("    \"finalClass\": " + c.getFinalClass() + ",");
                    writer.println("    \"methodType\": \"" + c.getMethodType() + "\",");
                    writer.println("    \"codeLength\": " + c.getCodeLength() + ",");
                    writer.println("    \"filePath\": \"" + c.getFilePath().replace("\\", "\\\\") + "\",");
                    writer.println("    \"createdAt\": \"" + c.getCreatedAt() + "\"");
                    writer.println("  }" + (i < candidates.size() - 1 ? "," : ""));
                }
                writer.println("]");
            }

            System.out.println("\nСохранено в: " + outputFile.toAbsolutePath());
            System.out.println("Всего сохранено примеров: " + candidates.size());

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}