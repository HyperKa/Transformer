package org.example.data;

public class ConfigConstants {
    // ОСНОВНАЯ СБОРКА
    /*
    public static final int NUM_LAYERS = 3;
    public static final int NUM_HEADS = 8;
    public static final int EMBEDDING_DIM = 64;
    public static final int FFN_HIDDEN_DIM = 256; // 64 * 4
    public static final int NUM_CLASSES = 8;

    public static final double LEARNING_RATE = 0.001;
    public static final int EPOCHS = 20;

    // Максимальная длина последовательности:
    public static final int MAX_LENGTH = 100;
    public static final String MODEL_SAVE_PATH = "C:/Users/VivoBook-15/Documents/Java/transformer/test_dataset.txt";
    */

    // ТЕСТОВАЯ СБОРКА
    public static final int NUM_LAYERS = 2;
    public static final int NUM_HEADS = 2;
    public static final int EMBEDDING_DIM = 4;
    public static final int FFN_HIDDEN_DIM = 16; // 64 * 4
    public static final int NUM_CLASSES = 2;

    public static final double LEARNING_RATE = 0.01;
    public static final int EPOCHS = 20;

    // Максимальная длина последовательности:
    public static final int MAX_LENGTH = 10;
    public static final String MODEL_SAVE_PATH = "C:/Users/VivoBook-15/Documents/Java/transformer/transformer_model.txt";
    public static final String VOCAB_SAVE_PATH = "C:/Users/VivoBook-15/Documents/Java/transformer/vocabulary.txt";
    public static final String MODEL_TRAINING_PATH = "C:/Users/VivoBook-15/Documents/Java/transformer/test_dataset2.txt";
    public static final String EXAMPLE_CODE_PATH = "C:/Users/VivoBook-15/Documents/Java/transformer/input_code.txt";
    public static final Long RANDOM_SEED = 42L;  // 42L
    public static final boolean IS_TEST = true;  // эти две переменные нужны для случайной генерации матриц, её контроля
}
