package org.example.parser.source;

import org.example.parser.model.CodeCandidate;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.zip.*;

public abstract class ZipArchiveLoader {

    protected final Path archivePath;
    protected final Path extractPath;

    public ZipArchiveLoader(String archivePath, String extractPath) throws IOException {
        this.archivePath = Paths.get(archivePath);
        this.extractPath = Paths.get(extractPath);
        extractIfNeeded();
    }

    /**
     * Распаковка архива, если он ещё не распакован
     */
    private void extractIfNeeded() throws IOException {
        if (Files.exists(extractPath) && hasJavaFiles(extractPath)) {
            System.out.printf("Архив уже распакован: %s%n", extractPath);
            return;
        }

        Files.createDirectories(extractPath);

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

        System.out.printf("Архив распакован: %s → %s%n", archivePath, extractPath);
    }

    private boolean hasJavaFiles(Path dir) throws IOException {
        try (var stream = Files.walk(dir)) {
            return stream.anyMatch(p -> p.toString().endsWith(".java"));
        }
    }

    /**
     * Поиск всех Java-файлов в распакованной директории
     */
    protected List<Path> findJavaFiles() throws IOException {
        List<Path> javaFiles = new ArrayList<>();
        try (var stream = Files.walk(extractPath)) {
            stream.filter(p -> p.toString().endsWith(".java"))
                    .forEach(javaFiles::add);
        }
        return javaFiles;
    }

    /**
     * Абстрактный метод для специфичной логики каждого источника
     */
    public abstract List<CodeCandidate> load() throws IOException;
}
