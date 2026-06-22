package org.example.parser.config;

import io.github.cdimascio.dotenv.Dotenv;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class MistralHeadersConfig {

    private static final List<String> KEYS;
    private static final AtomicInteger keyIndex = new AtomicInteger(0);

    static {
        String rawKeys = System.getenv("MISTRAL_KEYS");
        if (rawKeys == null || rawKeys.isEmpty()) {
            try {
                Dotenv dotenv = Dotenv.configure().ignoreIfMissing().load();
                rawKeys = dotenv.get("MISTRAL_KEYS");
            } catch (Exception e) {
                System.out.println(">>> .env файл не найден, работаю только с переменными окружения.");
            }
        }

        if (rawKeys != null && !rawKeys.isEmpty()) {
            KEYS = Arrays.stream(rawKeys.split(","))
                    .map(s -> s.replaceAll("[\\p{Cntrl}\\s]", ""))
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
            System.out.println(">>> УСПЕШНО: Загружено API ключей: " + KEYS.size());
        } else {
            KEYS = List.of();
            System.err.println(">>> ОШИБКА: Ключи MISTRAL_KEYS не найдены!");
        }
    }

    public static String getNextApiKey() {
        if (KEYS.isEmpty()) {
            throw new RuntimeException("API KEYS NOT FOUND!");
        }
        return KEYS.get(Math.abs(keyIndex.getAndIncrement() % KEYS.size()));
    }

    public static final String MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions";
    public static final String MISTRAL_MODEL = "open-mistral-7b";

    public static final String CLASS_DESCRIPTION =
            "0: SAFE (Безопасный код, недостижимый код, пустые безопасные методы)\n" +
                    "1: RESOURCE_LEAK (Утечки ресурсов: unclosed streams, sockets, database connections)\n" +
                    "2: BAD_LOGIC_AND_EXCEPTION (Пустые catch, забытые заглушки/stubs, деление на ноль, переполнение знаковых целых)\n" +
                    "3: XXE (Небезопасная конфигурация XML, отсутствие флагов защиты)\n" +
                    "4: NPE (NULL Pointer Dereference - обращение к объектам без проверки на null)\n" +
                    "5: RCE_AND_COMMANDS (Выполнение команд ОС через Runtime.exec или ProcessBuilder)\n" +
                    "6: INJECTION_GENERAL (SQL-инъекции, XPath, LDAP, небезопасные запросы)\n" +
                    "7: UNSAFE_API_AND_DESERIAL (Использование опасных функций API, рефлексии и десериализация объектов)";

    // 17 Специализированных ролей
    public static final Map<String, String> AGENT_ROLES = Map.ofEntries(
            // Защитники (Класс 0)
            Map.entry("H0_SAFE_A", "Ты защитник кода. Твоя задача — доказать, что код безопасен, экранирован или содержит проверку входных данных. Если явных угроз нет, голосуй за класс 0."),
            Map.entry("H0_SAFE_B", "Ты Java-архитектор. Оценивай структуру. Если код находится в тестовом окружении, является DTO или содержит защитную логику фреймворка (Spring/Hibernate) — ставь 0."),

            // Эксперты по утечкам (Класс 1)
            Map.entry("H1_LEAK_A", "Ты эксперт по утечкам ресурсов. Ищи незакрытые потоки ввода-вывода, сокеты, файлы или соединения с БД. Ставь класс 1 при их наличии."),
            Map.entry("H1_LEAK_B", "Ты системный аналитик. Проверяй сложные ветвления (if-else, try-catch). Если при ошибке поток выполнения выходит из метода без закрытия ресурсов — ставь класс 1."),

            // Эксперты по исключениям (Класс 2)
            Map.entry("H2_EXCEPT_A", "Ты аудитор надежности кода. Ищи пустые catch-блоки, забытые stubs, заглушки с null-переменными или закомментированную обработку ошибок. Ставь класс 2."),
            Map.entry("H2_EXCEPT_B", "Ты математический валидатор. Ищи критические ошибки бизнес-логики: деление на ноль, переполнение чисел (CWE-190) и некорректные касты типов. Ставь класс 2."),

            // Эксперты по XML (Класс 3)
            Map.entry("H3_XXE_A", "Ты эксперт по XML уязвимостям. Ищи инициализацию DocumentBuilderFactory, XMLInputFactory или SAXParser без защитных настроек. Ставь класс 3."),
            Map.entry("H3_XXE_B", "Ты специалист по безопасным конфигурациям XML. Проверяй, отключены ли внешние сущности (disallow-doctype-decl). Если нет — это XXE (класс 3)."),

            // Эксперты по NPE (Класс 4)
            Map.entry("H4_NPE_A", "Ты статический анализатор NPE. Ищи вызовы методов у объектов, которые могут быть null без предварительной проверки. Ставь класс 4."),
            Map.entry("H4_NPE_B", "Ты эксперт по потоку данных. Прослеживай цепочки вызовов. Если метод может вернуть null (или принимает null) и это ведет к разыменованию — ставь класс 4."),

            // Эксперты по RCE (Класс 5)
            Map.entry("H5_RCE_A", "Ты специалист по командным инъекциям. Ищи вызовы Runtime.getRuntime().exec() или ProcessBuilder с нефильтрованными аргументами. Ставь класс 5."),
            Map.entry("H5_RCE_B", "Ты системный администратор безопасности. Анализируй выполнение команд ОС. Если внешние параметры попадают в системный shell — ставь класс 5."),

            // Эксперты по SQLi (Класс 6)
            Map.entry("H6_SQL_A", "Ты пентестер баз данных. Ищи прямую конкатенацию строк в JDBC, JPA, Hibernate или SQL-запросах. Ставь класс 6."),
            Map.entry("H6_SQL_B", "Ты эксперт по taint-анализу. Прослеживай путь пользовательского ввода от аннотаций контроллера (@RequestParam) до SQL-запроса. Если нет экранирования — это класс 6."),

            // Эксперты по Unsafe API (Класс 7)
            Map.entry("H7_UNSAFE_A", "Ты криптограф. Ищи использование слабых алгоритмов (DES, MD5) и небезопасной рефлексии Java API. Ставь класс 7."),
            Map.entry("H7_UNSAFE_B", "Ты эксперт по десериализации. Ищи вызовы ObjectInputStream.readObject() без валидации типов. Это класс 7."),

            // Верховный судья
            Map.entry("H17_ARCHITECT", "Ты Верховный Архитектор. Твоя задача — разрешать конфликты между другими экспертами. Твое решение приоритетно в спорных ситуациях.")
    );
}