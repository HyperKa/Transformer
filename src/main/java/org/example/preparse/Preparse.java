package org.example.preparse;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class Preparse {
    public static void convert(String inputPath, String outputPath) throws IOException {
        try (Stream<String> lines = Files.lines(Paths.get(inputPath));
             BufferedWriter writer = new BufferedWriter(new FileWriter(outputPath))) {

            writer.write("[\n");

            // Используем итератор, чтобы отловить последний элемент
            var iterator = lines.iterator();
            while (iterator.hasNext()) {
                String line = iterator.next();
                writer.write(line);
                if (iterator.hasNext()) {
                    writer.write(",\n");
                } else {
                    writer.write("\n");

                }
            }

            writer.write("]\n");
        }
    }

    public static void main(String[] args) {
        try {
            // Укажите пути к вашим файлам
            convert("C:\\Users\\VivoBook-15\\Documents\\Java\\transformer\\datassets\\code-text-java\\train.jsonl",
                    "C:\\Users\\VivoBook-15\\Documents\\Java\\transformer\\datassets\\code-text-java\\train_formatted.json");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
