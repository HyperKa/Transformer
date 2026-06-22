package org.example.parser;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

public class UtilFixJuliet {
    private static final String DATASET_PATH = "final_training_dataset.txt";

    public static void patchJulietBraces(String DATASET_PATH) throws IOException {
        String content = Files.readString(Paths.get(DATASET_PATH));
        // Заменяем одну скобку перед меткой на две (метод + класс)
        // Важно: это сработает только если в конце Juliet ровно одна скобка
        String fixed = content.replace("}|||LABEL", "}\n}|||LABEL");
        Files.writeString(Paths.get(DATASET_PATH), fixed);
        System.out.println("Juliet исправлен!");
    }

    public static void main (String[] args) {
        try {
            patchJulietBraces(DATASET_PATH);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
