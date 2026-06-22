package org.example.parser.pipeline;

import org.example.parser.config.MistralHeadersConfig;
import org.example.parser.model.LLMVerdict;
import org.json.JSONArray;
import org.json.JSONObject;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class Stage2MistralClient {
    private final HttpClient client = HttpClient.newBuilder()
            .connectTimeout(Duration.ofSeconds(30))
            .build();

    public LLMVerdict sendRequest(String codeContext, String specificRole, String headName) {
        int maxRetries = 3;
        int attempt = 0;

        while (attempt < maxRetries) {
            try {
                // ФОРМИРУЕМ ЖЕСТКИЙ ПРОМПТ
                String fullSystemPrompt = specificRole + "\n" +
                        "ИНСТРУКЦИЯ КЛАССИФИКАЦИИ:\n" + MistralHeadersConfig.CLASS_DESCRIPTION + "\n" +
                        "ТВОЙ ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО В ФОРМАТЕ JSON: {\"class\": number, \"confidence\": number}\n" +
                        "НЕ ПИШИ НИКАКИХ ПОЯСНЕНИЙ. ТОЛЬКО JSON ОБЪЕКТ.";

                JSONObject payload = new JSONObject();
                payload.put("model", MistralHeadersConfig.MISTRAL_MODEL);
                payload.put("messages", new JSONArray()
                        .put(new JSONObject().put("role", "system").put("content", fullSystemPrompt))
                        .put(new JSONObject().put("role", "user").put("content", "ПРОАНАЛИЗИРУЙ КОД:\n" + codeContext))
                );

                // Гарантия JSON на уровне API
                payload.put("response_format", new JSONObject().put("type", "json_object"));
                payload.put("temperature", 0.1); // Минимальный шум

                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(MistralHeadersConfig.MISTRAL_API_URL))
                        .header("Content-Type", "application/json")
                        .header("Authorization", "Bearer " + MistralHeadersConfig.getNextApiKey())
                        .POST(HttpRequest.BodyPublishers.ofString(payload.toString()))
                        .build();

                HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

                if (response.statusCode() == 200) {
                    return parseResponse(response.body(), headName);
                } else if (response.statusCode() == 429) {
                    // Обработка лимитов бесплатного API
                    long sleepTime = 20000L;
                    System.out.println(headName + ": Rate limit (429). Ждем " + (sleepTime/1000) + " сек...");
                    Thread.sleep(sleepTime);
                } else {
                    System.err.println(headName + ": Ошибка API " + response.statusCode() + " -> " + response.body());
                }
            } catch (Exception e) {
                System.err.println(headName + ": Ошибка соединения: " + e.getMessage());
            }
            attempt++;
        }
        return new LLMVerdict(-1, 0, headName);
    }

    private LLMVerdict parseResponse(String body, String headName) {
        try {
            // распаковка ответа от сервера Mistral
            JSONObject responseJson = new JSONObject(body);
            String aiGeneratedText = responseJson.getJSONArray("choices")
                    .getJSONObject(0)
                    .getJSONObject("message")
                    .getString("content").trim();

            // удаление кавычек markdown, если прислан ответ json в них ```json { ... } ```
            if (aiGeneratedText.contains("```")) {
                aiGeneratedText = aiGeneratedText.replaceAll("```json|```", "").trim();
            }

            // парсинг JSON в объект
            JSONObject result = new JSONObject(aiGeneratedText);

            // регистрочувствительность, getInt извлекает value из key "class"
            int predictedClass = -1;
            if (result.has("class")) predictedClass = result.getInt("class");
            else if (result.has("Class")) predictedClass = result.getInt("Class");
            else if (result.has("label")) predictedClass = result.getInt("label");

            // Взятие уверенности, иначе 0
            double confRaw = result.optDouble("confidence", result.optDouble("Confidence", 0.0));

            // Приведение 0.95 -> 95 или сохранение 95 как есть
            int confidence = (confRaw <= 1.0 && confRaw > 0) ? (int)(confRaw * 100) : (int)confRaw;

            return new LLMVerdict(predictedClass, confidence, headName);

        } catch (Exception e) {
            // Если пришел совсем мусор (например, ошибка сервера или пустой ответ)
            System.err.println("\n[!] Ошибка парсинга " + headName + ". Ответ API: " + body);
            return new LLMVerdict(-1, 0, headName);
        }
    }
}