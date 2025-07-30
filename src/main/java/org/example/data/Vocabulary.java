package org.example.data;
import org.example.data.ConfigConstants;
import java.util.*;

public class Vocabulary {
    private final Map<String, Integer> wordToIndex;
    private final Map<Integer, String> indexToWord;
    private final int vocabSize;
    // Максимальная длина последовательности:
    // private final static int MAX_LENGTH = 100;
    // ТЕСТОВАЯ:
    private int maxLength = ConfigConstants.MAX_LENGTH;

    public Vocabulary(List<List<String>> allSequences) {
        this.wordToIndex = new HashMap<>();
        this.indexToWord = new HashMap<>();

        addToken("[PAD]", 0);
        addToken("[UNK]", 1);
        addToken("[SOS]", 2);
        addToken("[EOS]", 3);

        int counter = 4;
        for (List<String> strings : allSequences) {
            for (String token : strings) {
                if (!wordToIndex.containsKey(token)) {  // Если словарь НЕ содержит ключ "токен"
                    addToken(token, counter++);
                }
            }
        }
        this.vocabSize = wordToIndex.size();
    }

    public void addToken(String token, Integer index) {
        this.wordToIndex.put(token, index);
        this.indexToWord.put(index, token);
    }

    public int getVocabSize() {
        return this.vocabSize;
    }

    public PreparedData convertToIndexes(List<List<String>> wordSequences) {
        // массивы для передачи в PreparedData и дальнейшей связи с Main
        int[][] tokenMatrix = new int[wordSequences.size()][maxLength];
        int[][] attentionMask = new int[wordSequences.size()][maxLength];

        for (int i = 0; i < wordSequences.size(); i++) {
            List<String> vectorOfWords = wordSequences.get(i);

            // начало каждого примера в датасете со [SOS]
            tokenMatrix[i][0] = 2;  // [SOS] - токен старта строки, не помощь -_-, важная часть обучения, именно по нему идет корректировка остальных параметров, строка [1, 64] в 46-ой строке TransformerModel
            attentionMask[i][0] = 1;

            int realSequenceLength = Math.min(vectorOfWords.size(), maxLength - 2);

            for (int j = 0; j < realSequenceLength; j++) {
                String token = vectorOfWords.get(j);
                tokenMatrix[i][j + 1] = this.wordToIndex.getOrDefault(token, 1);  // [UNK] = 1, если слово не из словаря
                attentionMask[i][j + 1] = 1;
            }

            // добавление к каждому вектору в конец значения EOS
            System.out.println("Длина словаря для примера " + (i + 1) + " : " + realSequenceLength);
            int lastIndex = realSequenceLength + 1;
            tokenMatrix[i][lastIndex] = 3;  //  [EOS]
            attentionMask[i][lastIndex] = 1;
            // [PAD] есть неявно. Паддинг и маска имеют значение 0 в словаре при инициализации, если токен отсутствует до максимального значения, так что всё корректно
        }
        return new PreparedData(tokenMatrix, attentionMask);
    }


    public static class PreparedData {
        private final int[][] tokenMatrix;
        private final int[][] attentionMask;

        public PreparedData(int[][] tokenMatrix, int[][] attentionMask) {
            this.tokenMatrix = tokenMatrix;
            this.attentionMask = attentionMask;
        }

        public int[][] getTokenMatrix() {
            return this.tokenMatrix;
        }

        public int[][] getAttentionMask() {
            return this.attentionMask;
        }

        public void printTokenMatrix() {
            System.out.println("Вывод матрицы при инициализации: ");
            for (int i = 0; i < this.tokenMatrix.length; i++) {
                System.out.println(Arrays.toString(tokenMatrix[i]));
            }
        }

        public void printAttentionMask() {
            System.out.println("Вывод матрицы при инициализации: ");
            for (int i = 0; i < this.attentionMask.length; i++) {
                System.out.println(Arrays.toString(attentionMask[i]));
            }
        }

    }

    public int getDictionarySize() {
        return wordToIndex.size();
    }
}
