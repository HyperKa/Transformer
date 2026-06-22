package org.example;

import org.example.analysis.CodeAnalyzer;
import org.example.data.ConfigConstants;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Scanner;

public class Predict {

    private static final String[] CLASS_NAMES = {
            "Safe Code", "Resource Leak", "Bad Exception", "XXE",
            "NPE", "RCE", "SQLi", "Unsafe API"
    };

    public static void main(String[] args) throws IOException, ClassNotFoundException, IllegalAccessException {

        org.nd4j.linalg.factory.Nd4j.setDefaultDataTypes(
                org.nd4j.linalg.api.buffer.DataType.FLOAT,
                org.nd4j.linalg.api.buffer.DataType.FLOAT
        );

        CodeAnalyzer codeAnalyzer = new CodeAnalyzer("exported_model");
        Scanner scanner = new Scanner(System.in != null ? System.in : new java.io.ByteArrayInputStream(new byte[0]));

        // === АВТОТЕСТ 16 ПРИМЕРОВ ===
        runBatchTests(codeAnalyzer);

        analyzeFile(codeAnalyzer);
    }

    private static void runBatchTests(CodeAnalyzer codeAnalyzer) {
        TestCase[] tests = getExpertTestCases();
        System.out.println("\n================================== ТАБЛИЦА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ (16 КЕЙСОВ) ==================================");
        System.out.printf("%-15s | %-15s | %-15s | %-12s | %-10s\n", "ОЖИДАЛОСЬ", "СЫРОЙ ВЕРДИКТ", "ИТОГ (ГИБРИД)", "УВЕРЕННОСТЬ", "СТАТУС");
        System.out.println("------------------------------------------------------------------------------------------------------------------");

        for (TestCase tc : tests) {
            CodeAnalyzer.AnalysisResult res = codeAnalyzer.analyzeDetailed(tc.code);
            int rawPred = res.predictedClass;
            double confidence = res.confidence;

            // Применяем эвристику с учетом порога уверенности < 45%
            int finalPred = applyHeuristics(tc.code, rawPred, confidence);

            String status = (finalPred == tc.expectedLabel) ? "[OK]" : "[FAIL]";

            System.out.printf("%-15s | %-15s | %-15s | %-11.2f%% | %s\n",
                    CLASS_NAMES[tc.expectedLabel],
                    CLASS_NAMES[rawPred],
                    CLASS_NAMES[finalPred],
                    confidence * 100,
                    status);
        }
        System.out.println("==================================================================================================================\n");
    }

    private static int applyHeuristics(String code, int rawPrediction, double confidence) {
        if (confidence >= 0.45) {
            return rawPrediction; // Если нейросеть уверена — доверяем ей
        }
        String lower = code.toLowerCase();

        // RCE (захват сервера)
        if (lower.contains("processbuilder") || lower.contains("runtime.getruntime().exec")) {
            return 5;
        }

        // SQLi (компрометация БД)
        if (lower.contains("select") && lower.contains("'\" +")) {
            return 6;
        }

        // XXE (чтение файлов сервера)
        if (lower.contains("documentbuilderfactory") && !lower.contains("disallow-doctype-decl")) {
            return 3;
        }

        // Unsafe API (Высокая угроза выполнения кода)
        if (lower.contains("readobject") || lower.contains("unsafe.getunsafe")) {
            return 7;
        }

        // Resource Leak (отказ в обслуживании DoS)
        if (lower.contains("fileinputstream") && !lower.contains("try (")) {
            return 1;
        }

        // Bad Exception (логические ошибки обработки)
        if (lower.contains("catch") && (lower.contains("stub") || lower.contains("null") || lower.contains("todo") || lower.contains("placeholder") || lower.contains("err"))) {
            return 2;
        }

        return rawPrediction;
    }

    private static void showScenariosMenu() {
        System.out.println("\n--- СПИСОК РЕАЛЬНЫХ СЦЕНАРИЕВ ДЛЯ ГЕНЕРАЦИИ ---");
        System.out.println("0 - Безопасный Spring Boot сервис (Spring Data JPA, обработка ошибок)");
        System.out.println("1 - Resource Leak в Spring Service (незакрытый InputStream при парсинге CSV)");
        System.out.println("2 - Забытый null-заглушка stub в catch-блоке Spring бина");
        System.out.println("3 - XXE-инъекция в XmlParser с автосвязыванием конфигурации (Spring Component)");
        System.out.println("4 - Риск Null Pointer Exception (NPE) в Spring MVC Controller");
        System.out.println("5 - Выполнение команд ОС (RCE) через ProcessBuilder в REST-контроллере");
        System.out.println("6 - SQL-инъекция в Spring RestController через конкатенацию в JdbcTemplate");
        System.out.println("7 - Небезопасная десериализация через Java-импорт устаревшего API");
    }


    private static void writeCodeToFile(String code) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(Paths.get("input_code.txt"))) {
            writer.write(code);
        }
    }

    private static void analyzeFile(CodeAnalyzer codeAnalyzer) {
        try {
            BufferedReader reader = Files.newBufferedReader(Paths.get("input_code.txt"));
            StringBuilder stringBuilder = new StringBuilder();
            String currentLine;
            while ((currentLine = reader.readLine()) != null) {
                stringBuilder.append(currentLine);
                stringBuilder.append(System.lineSeparator());
            }

            String code = stringBuilder.toString();
            System.out.println("\n================ КОД ДЛЯ АНАЛИЗА ================");
            System.out.println(code);
            System.out.println("=================================================");

            CodeAnalyzer.AnalysisResult res = codeAnalyzer.analyzeDetailed(code);
            int rawPred = res.predictedClass;
            double confidence = res.confidence;

            int finalPred = applyHeuristics(code, rawPred, confidence);

            System.out.printf("\n>>> УВЕРЕННОСТЬ НЕЙРОСЕТИ: %.2f%%\n", confidence * 100);
            System.out.print(">>> ВЕРДИКТ НЕЙРОСЕТИ (С УЧЕТОМ СТАТИЧЕСКИХ ПРАВИЛ): ");
            switch (finalPred) {
                case 0:
                    System.out.println("Класс 0: Код, скорее всего, безопасный.");
                    break;
                case 1:
                    System.out.println("Класс 1: Обнаружена утечка ресурсов (Resource Leak).");
                    break;
                case 2:
                    System.out.println("Класс 2: Обнаружена некорректная обработка исключений (заглушки/пустые catch).");
                    break;
                case 3:
                    System.out.println("Класс 3: Обнаружена XXE-инъекция.");
                    break;
                case 4:
                    System.out.println("Класс 4: Обнаружен риск Null Pointer Exception (NPE).");
                    break;
                case 5:
                    System.out.println("Класс 5: Обнаружено удаленное выполнение кода (RCE).");
                    break;
                case 6:
                    System.out.println("Класс 6: Обнаружена SQL-инъекция.");
                    break;
                case 7:
                    System.out.println("Класс 7: Использование устаревшего или небезопасного API / Десериализация.");
                    break;
                default:
                    System.out.println("Неизвестный класс уязвимости: " + finalPred);
                    break;
            }
            System.out.println();

        } catch (IOException e) {
            System.out.println("[-] Произошла ошибка во время чтения файла");
            e.printStackTrace();
        }
    }

    private static TestCase[] getExpertTestCases() {
        return new TestCase[]{
                new TestCase("public void processData(String payload) {\n" +
                        "    if (payload == null || payload.isEmpty()) return;\n" +
                        "    int tempCounter = 0;\n" +
                        "    for(int i=0; i < payload.length(); i++) tempCounter++;\n" +
                        "    System.out.println(\"Processed: \" + payload.trim() + \" count: \" + tempCounter);\n" +
                        "}", 0),

                new TestCase("public void secureUpdate(Connection db, int userId, String status) throws SQLException {\n" +
                        "    String sqlStr = \"UPDATE registry SET status = ? WHERE id = ?\";\n" +
                        "    try (PreparedStatement ps = db.prepareStatement(sqlStr)) {\n" +
                        "        ps.setString(1, status);\n" +
                        "        ps.setInt(2, userId);\n" +
                        "        ps.executeUpdate();\n" +
                        "    }\n" +
                        "}", 0),

                new TestCase("public void exportInternal(String targetPath) throws Exception {\n" +
                        "    FileOutputStream fos = new FileOutputStream(targetPath);\n" +
                        "    boolean isEmergency = checkSystemStatus();\n" +
                        "    if (isEmergency) return; \n" +
                        "    fos.write(new byte[]{1, 2, 3});\n" +
                        "    fos.close();\n" +
                        "}", 1),

                new TestCase("public void notifyServer(String addr) throws IOException {\n" +
                        "    Socket socket_conn = new Socket(addr, 443);\n" +
                        "    socket_conn.getOutputStream().write(255);\n" +
                        "    socket_conn.close();\n" +
                        "}", 1),

                new TestCase("public void syncState() {\n" +
                        "    try { \n" +
                        "        remoteSvc.reboot(); \n" +
                        "    } catch (Exception err) {}\n" +
                        "}", 2),

                new TestCase("public int computeOffset(int base, int multiplier) {\n" +
                        "    System.out.println(\"Debug log: starting calc...\");\n" +
                        "    return base * multiplier + 1024;\n" +
                        "}", 2),

                new TestCase("public void handleXml(InputStream blob) throws Exception {\n" +
                        "    DocumentBuilderFactory factory_instance = DocumentBuilderFactory.newInstance();\n" +
                        "    System.out.println(\"Factory ready: \" + factory_instance.getClass().getName());\n" +
                        "    DocumentBuilder parser = factory_instance.newDocumentBuilder();\n" +
                        "    parser.parse(blob); \n" +
                        "}", 3),

                new TestCase("public void parseStream(Reader rdr) throws Exception {\n" +
                        "    XMLInputFactory stax_f = XMLInputFactory.newInstance();\n" +
                        "    XMLStreamReader x_reader = stax_f.createXMLStreamReader(rdr);\n" +
                        "}", 3),

                new TestCase("public void updateNode(User u) {\n" +
                        "    if (u == null) {\n" +
                        "        logger.error(\"Node user is missing!\");\n" +
                        "    }\n" +
                        "    String nodeName = u.getName(); \n" +
                        "    System.out.println(\"Node: \" + nodeName);\n" +
                        "}", 4),

                new TestCase("public void showAvatar(Profile p) {\n" +
                        "    String url = p.getSettings().getAvatar().getUrl(); \n" +
                        "    render(url); \n" +
                        "}", 4),

                new TestCase("public void callSystem(String x_param) throws IOException {\n" +
                        "    String binary_path = \"/usr/bin/service_manager\";\n" +
                        "    Runtime.getRuntime().exec(binary_path + \" --cmd \" + x_param); \n" +
                        "}", 5),

                new TestCase("public void initProc(List<String> list_of_commands) throws IOException {\n" +
                        "    ProcessBuilder builder_obj = new ProcessBuilder(list_of_commands);\n" +
                        "    builder_obj.start(); \n" +
                        "}", 5),

                new TestCase("public void searchRegistry(String search_term) throws SQLException {\n" +
                        "    int loggingLevel = 1;\n" +
                        "    String raw_query = \"SELECT * FROM items WHERE tag = '\" + search_term + \"'\";\n" +
                        "    if(loggingLevel > 0) System.out.println(\"Exec query...\");\n" +
                        "    db_connection.createStatement().execute(raw_query); \n" +
                        "}", 6),

                new TestCase("public void findInLogs(String pattern) {\n" +
                        "    String format_str = \"SELECT msg FROM app_logs WHERE msg LIKE '%%%s%%'\";\n" +
                        "    String final_sql = String.format(format_str, pattern);\n" +
                        "    db.run(final_sql); \n" +
                        "}", 6),

                new TestCase("public Object restoreObject(byte[] raw_bytes) throws Exception {\n" +
                        "    ObjectInputStream stream_in = new ObjectInputStream(new ByteArrayInputStream(raw_bytes));\n" +
                        "    return stream_in.readObject(); \n" +
                        "}", 7),

                new TestCase("public void memoryHack(long base_addr) {\n" +
                        "    sun.misc.Unsafe low_level_api = Unsafe.getUnsafe();\n" +
                        "    low_level_api.putLong(base_addr, 0L); \n" +
                        "}", 7)
        };
    }

    private static class TestCase {
        public final String code;
        public final int expectedLabel;

        public TestCase(String code, int expectedLabel) {
            this.code = code;
            this.expectedLabel = expectedLabel;
        }
    }
}