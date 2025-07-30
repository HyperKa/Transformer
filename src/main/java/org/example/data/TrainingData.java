package org.example.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class TrainingData {

    public TrainingData(String filePath) throws IOException {
        this.rawData = new String(Files.readAllBytes(Paths.get(filePath)));
        parse3();
    }
    // private String trainingMas = "// --- safety ---\n" + "}|||LABEL|||0\n" + "\n" + "// --- Using ---\n" + "class UserConvert {\n" + "    void convert" + "    }\n" + "\n" + "}|||LABEL|||0";
    private String trainingMas = "// --- Using try-with-resources properly ---\n" +
            "class SafeFileReader {\n" +
            "    public void read(String path) throws IOException {\n" +
            "        try (FileInputStream fis = new FileInputStream(path)) {\n" +
            "            int data;\n" +
            "            while ((data = fis.read()) != -1) {\n" +
            "                System.out.print((char) data);\n" +
            "            }\n" +
            "        }\n" +
            "    }\n" +
            "}|||LABEL|||0\n" +
            "\n" +
            "// --- Using prepared statement safely ---\n" +
            "class SafeSQLQuery {\n" +
            "    void query(Connection conn, String userInput) {\n" +
            "        try {\n" +
            "            String query = \"SELECT * FROM users WHERE username = ?\";\n" +
            "            try (PreparedStatement stmt = conn.prepareStatement(query)) {\n" +
            "                stmt.setString(1, userInput);\n" +
            "                ResultSet rs = stmt.executeQuery();\n" +
            "                while (rs.next()) {\n" +
            "                    System.out.println(rs.getString(\"username\"));\n" +
            "                }\n" +
            "            }\n" +
            "        } catch (SQLException e) {\n" +
            "            e.printStackTrace();\n" +
            "        }\n" +
            "    }\n" +
            "}|||LABEL|||0";
    private String trainingMas2 = trainingMas;
    private List<String> codeSnippets = new ArrayList<>();
    private List<Integer> labels = new ArrayList<>();
    private String rawData;


    public ArrayList<ArrayList<String>> parse() {

        trainingMas = trainingMas.replaceAll("[^a-zA-Zа-яА-Я0-9@=+\\-*/|;.(){}\n ]", " ")
                    .toLowerCase();

        ArrayList<ArrayList<String>> newVector = new ArrayList<>();  // вектор всех примеров датасета
        ArrayList<String> localVector = new ArrayList<>();           // вектор конкретного примера
        boolean flag_comment = false;

        // trainingMas = trainingMas.split("(?<=[(){};,.])|(?=[(){};,.])");

        for (String word1 : trainingMas.split("\s+|(?<=[(){};,.])|(?=[(){};,.])")) {          // разделить по одному или более пробельному символу

            if (word1.isEmpty() || word1.equals(" ")) {                        // Фильтр от пробелов, чтобы не тратить время обработки
                continue;
            }

            if (word1.equals("//") || word1.equals("/*")) {                    // включаю флаг игнорирования комментариев
                flag_comment = true;
            }

            if (word1.contains("\n")) {
                if (flag_comment) {
                    word1 = word1.substring(word1.indexOf("\n") + 1);
                    flag_comment = false;
                    if (word1.isEmpty()) {
                        continue;
                    }
                }
                String[] secondLocalList = word1.split("\n+");

                for (String word : secondLocalList) {
                    if (word.equals("//")) {
                        flag_comment = true;
                        continue;
                    }
                    localVector.add(word);
                }
            }
            else {
                if (!flag_comment) {
                    localVector.add(word1);
                }
            }
            if (word1.startsWith("|||label|||")) {                 // Разделитель примеров
                newVector.add(new ArrayList<>(localVector));
                System.out.println("Сработало добавление вектора в вектор");
                localVector.clear();
            }
        }
        return newVector;
    }


    public ArrayList<ArrayList<String>> parse2() {
        ArrayList<ArrayList<String>> mainVector = new ArrayList<>();
        String[] finalSequence = trainingMas2.split("\\|\\|\\|LABEL\\|\\|\\|\\d+");
        String noBlockComments, noComments;

        for (String part : finalSequence) {
            if (part.isEmpty()) {
                continue;
            }

            noBlockComments = part.replaceAll("(?s)/\\*.*?\\*/", "");  // (?s) - режим DOTALL для игнорирования '\n' в нежадном поиске
            noComments = noBlockComments.replaceAll("//.*", "");
            ArrayList<String> localList = new ArrayList<>();
            String[] cleanCode = noComments.split("\\s+|(?<=[{}();,.])|(?=[{}();,.])|\"");

            for (String insertPart : cleanCode) {
                if (!insertPart.isEmpty()) {
                    localList.add(insertPart);
                }
            }
            if (!localList.isEmpty()) {
                mainVector.add(localList);
            }

        }
        return mainVector;
    }

    private void parse3() {
        this.codeSnippets = new ArrayList<>();
        this.labels = new ArrayList<>();

        // Паттерн, который находит и код (группа 1), и метку (группа 2), является ШАБЛОНОМ для разделения примеров в датасете
        Pattern pattern = Pattern.compile("(.+?)\\|\\|\\|LABEL\\|\\|\\|(\\d+)", Pattern.DOTALL);  // ".+?" - любой текст + LABEL и "\\d+" - любое число, DOTALL - позволяет точке совпадать и со спецсимволом "\n". В pattern находится СТРОКА вида "<код>|||LABEL|||<число>". Грубо говоря, это разделение по |||LABEL||| на две части, где группы выделяются в виде последовательностей в скобках "(...)"
        // Matcher не является структурой с полями, ничего в себе не хранит глобально. Он только обрабатывает текущий пример, найденный по указателю с методом .find(), поэтому проход по всему датасету с объектом matcher происходит циклически с .find(). Группы выделяются также по указателю
        Matcher matcher = pattern.matcher(this.rawData);  // сопоставление примеров по паттерну, один пример = одно совпадение подстроки в формате паттерна. Запись в структуру Matcher списка паттернов и всей строки rawData

        while (matcher.find()) {  // перебор всех совпадений
            String codeBlock = matcher.group(1).trim();  // сохранение всей части до LABEL. group(0) хранит в себе всю строку rawData. trim() удаляет все пробелы в начале и в конце строки
            int label = Integer.parseInt(matcher.group(2));  // сохранение значения метки LABEL

            this.codeSnippets.add(codeBlock);
            this.labels.add(label);
        }
    }

    public List<String> getCodeSnippets() { return this.codeSnippets; }

    public List<String> tokenizeCode(String codeBlock) {
        String noBlockComments = codeBlock.replaceAll("(?s)/\\*.*?\\*/", "");
        String noComments = noBlockComments.replaceAll("//.*", "");
        List<String> tokens = new ArrayList<>();
        String[] cleanCode = noComments.split("\\s+|(?<=[{}();,.])|(?=[{}();,.])|\"");

        for (String token : cleanCode) {
            if (!token.isEmpty()) {
                tokens.add(token);
            }
        }
        return tokens;
    }

    public List<Integer> extractLabels() {
        if (this.labels == null) {
            this.labels = new ArrayList<>();
            Pattern pattern = Pattern.compile("\\|\\|\\|LABEL\\|\\|\\|(\\d+)");
            Matcher matcher = pattern.matcher(this.trainingMas2);

            while(matcher.find()) {
                // group(1) - это то, что попало в круглые скобки (\\d+)
                int label = Integer.parseInt(matcher.group(1));
                this.labels.add(label);
            }
        }
        return this.labels;
    }

    public List<Integer> getLabels() {
        if (this.labels == null) {
            return extractLabels();
        }
        return this.labels;
    }
}
