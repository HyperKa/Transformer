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
        // 1. Пытаемся загрузить из системы (для Docker)
        String rawKeys = System.getenv("MISTRAL_KEYS");

        // 2. Если в системе пусто, пытаемся прочитать .env файл (для IDE)
        if (rawKeys == null || rawKeys.isEmpty()) {
            try {
                Dotenv dotenv = Dotenv.configure().ignoreIfMissing().load();
                rawKeys = dotenv.get("MISTRAL_KEYS");
            } catch (Exception e) {
                System.out.println(">>> .env файл не найден, работаю только с переменными окружения.");
            }
        }

        // 3. Парсим ключи
        if (rawKeys != null && !rawKeys.isEmpty()) {
            KEYS = Arrays.stream(rawKeys.split(","))
                    .map(s -> s.replaceAll("[\\p{Cntrl}\\s]", "")) // Чистим от \r, \n и пробелов
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toList());
            System.out.println(">>> УСПЕШНО: Загружено API ключей: " + KEYS.size());
        } else {
            KEYS = List.of();
            System.err.println(">>> ОШИБКА: Ключи MISTRAL_KEYS не найдены ни в системе, ни в .env!");
        }
    }

    public static String getNextApiKey() {
        if (KEYS.isEmpty()) {
            throw new RuntimeException("API KEYS NOT FOUND! Проверьте файл .env или переменные Docker.");
        }
        return KEYS.get(Math.abs(keyIndex.getAndIncrement() % KEYS.size()));
    }

    public static final String MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions";
    public static final String MISTRAL_MODEL = "open-mistral-7b";


    // Описание с международными метками CWE
    public static final String CLASS_DESCRIPTION =
            "0: SAFE (CWE-561, CWE-570/571: Безопасный код, недостижимый код, пустые, но безопасные методы)\n" +
                    "1: RESOURCE_LEAK (CWE-404, CWE-772, CWE-459: Утечки ресурсов: потоки, сокеты, соединения с БД)\n" +
                    "2: BAD_LOGIC_AND_EXCEPTION (CWE-248, CWE-460, CWE-190, CWE-369: Ошибки обработки ошибок, пустые catch, переполнение чисел, деление на ноль)\n" +
                    "3: XXE (CWE-611, CWE-827: Небезопасная конфигурация XML, внешние сущности)\n" +
                    "4: NPE (CWE-476: NULL Pointer Dereference - отсутствие проверок на null перед использованием)\n" +
                    "5: RCE_AND_COMMANDS (CWE-78, CWE-94, CWE-114: Выполнение команд ОС, внедрение кода через Runtime.exec или ProcessBuilder)\n" +
                    "6: INJECTION_GENERAL (CWE-89, CWE-90, CWE-643: Все виды инъекций в запросы: SQL, LDAP, XPath)\n" +
                    "7: UNSAFE_API_AND_DESERIAL (CWE-676, CWE-502: Использование опасных функций Java API и небезопасная десериализация объектов)";

    public static final Map<String, String> HEAD_PROMPTS = Map.of(
            "H1_STRICT", "Ты педантичный аудитор безопасности. Твоя база знаний - стандарты CWE и OWASP. Ищи только критические нарушения.",
            "H2_ARCHITECT", "Ты Java Architect. Твоя цель - снизить False Positives. Если код защищен или находится в тестовом окружении - ставь 0.",
            "H3_HACKER", "Ты этичный хакер. Твоя задача - найти вектор атаки. Если CWE-89 или CWE-78 возможны - классифицируй их.",
            "H4_RESOURCES", "Ты спец по системным ресурсам. Ищи нарушения CWE-404 (ресурсы) и CWE-476 (NPE).",
            "H5_INTUITION", "Ты Senior Developer. Твоя интуиция нацелена на поиск 'грязного' и небезопасного кода по стандартам CWE.",
            "H6_EXCEPTIONS", "Ты инженер по надежности. Ищи слабые места в обработке исключений согласно CWE-248 и CWE-460.",
            "H7_SANITIZER", "Ты эксперт по фильтрации данных. Проверяй, нейтрализован ли ввод перед попаданием в Sink-функции CWE."
    );
}