package org.example.migration;

import org.example.data.ConfigConstants;
import org.example.data.Vocabulary;
import org.example.model.TransformerModel;

import java.io.IOException;

public class MigrateApp {
    public static void main(String[] args) {
        // Папка, которую вы скопировали в корень проекта (содержит .npy и vocab.json)
        String weightsDir = "exported_model";

        try {
            System.out.println("=== СТАРТ МИГРАЦИИ МОДЕЛИ ИЗ PYTORCH В ND4J ===");

            // 1. Импортируем новый словарь из JSON
            System.out.println("\n[Шаг 1/4] Импорт словаря из JSON...");
            Vocabulary vocabulary = Vocabulary.loadFromJson(weightsDir + "/vocab.json");

            // 2. Инициализируем пустую структуру TransformerModel с размерами из ConfigConstants
            System.out.println("\n[Шаг 2/4] Создание пустой структуры TransformerModel в ND4J...");
            TransformerModel model = new TransformerModel(
                    vocabulary.getDictionarySize(),
                    ConfigConstants.EMBEDDING_DIM, // 256
                    ConfigConstants.NUM_HEADS,     // 8
                    ConfigConstants.FFN_HIDDEN_DIM, // 1024
                    ConfigConstants.NUM_LAYERS,    // 6
                    ConfigConstants.NUM_CLASSES    // 8
            );

            // 3. Переносим распакованные веса из .npy файлов в память ND4J модели
            System.out.println("\n[Шаг 3/4] Заполнение модели весами из .npy...");
            ModelWeightLoader.loadWeights(model, weightsDir);

            // 4. Сериализуем и сохраняем модель и словарь в файлы .ser
            System.out.println("\n[Шаг 4/4] Сохранение готовой модели в файлы .ser...");
            model.save(ConfigConstants.MODEL_SAVE_PATH);
            vocabulary.save(ConfigConstants.VOCAB_SAVE_PATH);

            System.out.println("\n[+] УСПЕХ: Файлы " + ConfigConstants.MODEL_SAVE_PATH +
                    " и " + ConfigConstants.VOCAB_SAVE_PATH + " готовы к работе!");

        } catch (IOException | IllegalAccessException e) {
            System.err.println("\n[!] КРИТИЧЕСКАЯ ОШИБКА МИГРАЦИИ:");
            e.printStackTrace();
        }
    }
}