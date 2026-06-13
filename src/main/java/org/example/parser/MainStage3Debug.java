package org.example.parser;

import org.example.parser.model.CodeCandidate;
import org.example.parser.model.LLMVerdict;
import org.example.parser.model.SonarVerdict;
import org.example.parser.pipeline.*;

import java.util.ArrayList;
import java.util.List;

public class MainStage3Debug {
    public static void main(String[] args) {
        System.out.println("=== ГЛОБАЛЬНЫЙ ТЕСТ: 16 РЕАЛЬНЫХ СЦЕНАРИЕВ (КЛАССЫ 0-7) ===\n");

        List<CodeCandidate> batch = createTestBatch();

        Stage2ClassificationManager aiManager = new Stage2ClassificationManager();
        Stage3SonarVerifier sonarScanner = new Stage3SonarVerifier();

        for (CodeCandidate candidate : batch) {
            System.out.println("\n>>> ПРОВЕРКА: " + candidate.getSource());
            System.out.println("-------------------------------------------------------");

            // 1. АНАЛИЗ ИИ
            List<LLMVerdict> verdicts = aiManager.classify(candidate);
            MistralArbitrator.AIResult ai = MistralArbitrator.getAnalysis(verdicts);

            // 2. АНАЛИЗ ЛИНТЕРА
            SonarVerdict sonar = sonarScanner.analyze(candidate);

            // 3. ФИНАЛЬНЫЙ АРБИТР
            SonarArbitrator.ArbitrationResult result = SonarArbitrator.determineFinalLabel(ai, sonar);

            // ВЫВОД АНАЛИТИКИ (Исправлено!)
            int winner = ai.getWinner();
            double winnerWeight = ai.getWeight(winner);

            System.out.println("AI TOP-1:  Класс " + winner + " (Вес: " + String.format("%.2f", winnerWeight) + ")");
            System.out.println("AI TOP-2:  " + ai.getTopN(2));
            System.out.println("SONAR:     Класс " + sonar.getPredictedClass() + " (" + sonar.getRuleName() + ")");

            System.out.println("\nРЕЗУЛЬТАТ АРБИТРАЖА:");
            System.out.println("  СТАТУС:    [" + result.status + "]");
            System.out.println("  ИТОГ:      |||LABEL|||" + result.label);
            System.out.println("-------------------------------------------------------");
        }

        aiManager.shutdown();
    }

    private static List<CodeCandidate> createTestBatch() {
        List<CodeCandidate> batch = new ArrayList<>();

        // --- КЛАСС 0: SAFE ---
        batch.add(create("C0_SAFE_JDBC", "public void save(Connection c, int id) throws SQLException {\n" +
                "    String sql = \"UPDATE users SET last_login = NOW() WHERE id = ?\";\n" +
                "    try (PreparedStatement ps = c.prepareStatement(sql)) {\n" +
                "        ps.setInt(1, id);\n" +
                "        ps.executeUpdate();\n" +
                "    }\n}"));

        batch.add(create("C0_SAFE_IO", "public String read(Path p) throws IOException {\n" +
                "    if (!Files.exists(p)) return null;\n" +
                "    try (BufferedReader br = Files.newBufferedReader(p)) {\n" +
                "        return br.readLine();\n" +
                "    }\n}"));

        // --- КЛАСС 1: RESOURCE_LEAK ---
        batch.add(create("C1_LEAK_FILE", "public void leakFile(String path) throws Exception {\n" +
                "    FileInputStream fis = new FileInputStream(path);\n" +
                "    int data = fis.read();\n" +
                "    if (data == -1) return; // УТЕЧКА: метод вышел, поток открыт\n" +
                "    fis.close();\n}"));

        batch.add(create("C1_LEAK_SOCKET", "public void send(String ip) throws IOException {\n" +
                "    Socket s = new Socket(ip, 80);\n" +
                "    s.getOutputStream().write(1);\n" +
                "    // НЕТ try-finally: если write упадет, сокет не закроется\n" +
                "    s.close();\n}"));

        // --- КЛАСС 2: BAD_LOGIC / OVERFLOW ---
        batch.add(create("C2_BAD_CATCH", "public void process() {\n" +
                "    try { doWork(); } catch (Exception e) {\n" +
                "        // Игнорируем ошибку (S112)\n" +
                "    }\n}"));

        batch.add(create("C2_OVERFLOW", "public int calc(int a, int b) {\n" +
                "    log.info(\"Calculating sum...\");\n" +
                "    return a + b; // CWE-190: риск переполнения int\n}"));

        // --- КЛАСС 3: XXE ---
        batch.add(create("C3_XXE_DBF", "public void parse(InputStream is) throws Exception {\n" +
                "    DocumentBuilderFactory f = DocumentBuilderFactory.newInstance();\n" +
                "    DocumentBuilder b = f.newDocumentBuilder(); // S2755: нет защиты\n" +
                "    b.parse(is);\n}"));

        batch.add(create("C3_XXE_STAX", "public void readXml(Reader r) throws Exception {\n" +
                "    XMLInputFactory f = XMLInputFactory.newInstance();\n" +
                "    XMLStreamReader reader = f.createXMLStreamReader(r);\n}"));

        // --- КЛАСС 4: NPE ---
        batch.add(create("C4_NPE_DEREF", "public void printUser(User u) {\n" +
                "    if (u == null) log.error(\"Null user!\");\n" +
                "    System.out.println(u.getName()); // S2259: u может быть null\n}"));

        batch.add(create("C4_NPE_MAP", "public void find(String key) {\n" +
                "    Object val = myMap.get(key);\n" +
                "    System.out.println(val.hashCode()); // Риск NPE\n}"));

        // --- КЛАСС 5: RCE ---
        batch.add(create("C5_RCE_EXEC", "public void runCmd(String cmd) throws IOException {\n" +
                "    Runtime.getRuntime().exec(\"sh -c \" + cmd); // S2076: Инъекция команд\n}"));

        batch.add(create("C5_RCE_PB", "public void start(List<String> args) throws IOException {\n" +
                "    ProcessBuilder pb = new ProcessBuilder(args);\n" +
                "    pb.start();\n}"));

        // --- КЛАСС 6: SQL_INJECTION ---
        batch.add(create("C6_SQL_RAW", "public void query(String user) throws SQLException {\n" +
                "    Statement st = conn.createStatement();\n" +
                "    st.executeQuery(\"SELECT * FROM users WHERE name = '\" + user + \"'\");\n}"));

        batch.add(create("C6_SQL_TAINTED", "public void search(String term) {\n" +
                "    String query = \"SELECT data FROM table WHERE info = \" + term;\n" +
                "    db.execute(query); // Линтер должен увидеть tainted переменную\n}"));

        // --- КЛАСС 7: UNSAFE_API ---
        batch.add(create("C7_DESERIAL", "public Object load(byte[] b) throws Exception {\n" +
                "    ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(b));\n" +
                "    return ois.readObject(); // Класс 7: Опасная десериализация\n}"));

        batch.add(create("C7_UNSAFE", "public void hack() {\n" +
                "    sun.misc.Unsafe u = Unsafe.getUnsafe();\n" +
                "    u.allocateInstance(Object.class);\n}"));

        return batch;
    }

    private static CodeCandidate create(String source, String code) {
        CodeCandidate c = new CodeCandidate();
        c.setSource(source);
        c.setFullContext("package org.test;\nimport java.util.*;\nimport java.io.*;\nimport java.sql.*;\nclass TestClass {\n" + code + "\n}");
        c.setVulnerableMethod("testMethod");
        return c;
    }
}