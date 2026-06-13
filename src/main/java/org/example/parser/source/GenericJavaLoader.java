package org.example.parser.source;

import org.example.parser.model.CodeCandidate;
import org.example.parser.pipeline.Stage1AstExtractor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class GenericJavaLoader extends ZipArchiveLoader {

    // Имя проекта, из которого извлечён файл
    private final String sourceName;

    public GenericJavaLoader(String archivePath, String extractPath, String sourceName) throws IOException {
        super(archivePath, extractPath);
        this.sourceName = sourceName;
    }

    @Override
    public List<CodeCandidate> load() throws IOException {
        List<CodeCandidate> candidates = new ArrayList<>();
        List<Path> javaFiles = findJavaFiles();

        System.out.printf("%s: найдено %d Java-файлов%n", sourceName, javaFiles.size());

        // Файл (file) - это полный путь до файла ("elasticsearch/src/org/elasticsearch/index/IndexService.java")
        for (Path file : javaFiles) {
            String content = Files.readString(file);

            // быстрый поиск подозрительных файлов по содержимому - ключевым словам
            if (hasDangerousPattern(content)) {
                List<CodeCandidate> extracted = Stage1AstExtractor.extractCandidates(content, sourceName, file.toString());
                candidates.addAll(extracted);
            }
        }

        System.out.printf("%s: отобрано %d кандидатов с контекстом%n", sourceName, candidates.size());
        return candidates;
    }

    // функция-пометка подозрительных методов
    /**
     * Цель: отбросить чистые DTO, Enums и примитивные утилиты.
     * Если файл содержит хоть один намек на работу с БД, сетью, файлами или исключениями - пропуск его к AST-парсеру.
     */
    private boolean hasDangerousPattern(String content) {
        String lowerContent = content.toLowerCase();

        String[] fileLevelKeywords = {
                // Класс 6: SQL (JDBC, JPA, Hibernate)
                "sql", "jdbc", "query", "statement", "entitymanager", "repository",

                // Класс 5: RCE (OS Commands)
                "exec", "process", "runtime", "bash", "cmd",

                // Класс 1: Resource Leak (Потоки, Файлы, Сеть, БД)
                "stream", "socket", "file", "reader", "writer", "connection", "url",

                // Класс 3: XXE (XML парсеры)
                "xml", "documentbuilder", "sax", "parser",

                // Класс 7: Unsafe API & Deserialization
                "unsafe", "readobject", "invoke",

                // Класс 2: Исключения
                "catch", "throw", "exception",

                // Класс 4: NPE
                "null",

                // Точки входа (Web, API)
                "mapping", "request", "response", "http"
        };

        for (String keyword : fileLevelKeywords) {
            if (lowerContent.contains(keyword)) {
                return true;
            }
        }

        return false; // Файл нормальный, интерфейс или DTO
    }
}