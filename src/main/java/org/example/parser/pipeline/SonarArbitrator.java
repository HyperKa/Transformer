package org.example.parser.pipeline;

import org.example.parser.model.SonarVerdict;
import org.example.parser.pipeline.MistralArbitrator.AIResult;

public class SonarArbitrator {

    public static class ArbitrationResult {
        public final int label;
        public final String status;

        public ArbitrationResult(int label, String status) {
            this.label = label;
            this.status = status;
        }
    }

    public static ArbitrationResult determineFinalLabel(AIResult ai, SonarVerdict sonar) {
        int sonarClass = sonar.getPredictedClass();
        int aiWinner = ai.getWinner();
        double aiWinnerWeight = ai.getWeight(aiWinner);
        double aiSafeWeight = ai.getWeight(0);
        double aiSupportForSonar = ai.getWeight(sonarClass);

        // 1. ВЕТО (Высший приоритет)
        // Если ИИ-лидер говорит SAFE и его вес значителен (> 1.2)
        // Мы верим ИИ и отменяем ЛЮБЫЕ подозрения линтера.
        if (aiWinner == 0 && aiSafeWeight > 1.2 && sonarClass != 0) {
            return new ArbitrationResult(0, "ВЕТО (AI REJECT)");
        }

        // 2. СИНЕРГИЯ (Строгая: только Топ-2)
        // Условие: Сонар нашел Х, и этот Х входит в ТОП-2 ИИ,
        // И ХОТЯ БЫ ОДНА голова ИИ его поддержала (вес > 0.8)
        if (sonarClass != 0 && ai.getTopN(2).contains(sonarClass) && aiSupportForSonar >= 0.8) {
            return new ArbitrationResult(sonarClass, "СИНЕРГИЯ");
        }

        // 3. ОТКРЫТИЕ (Когда линтер молчит)
        if (sonarClass == 0 && aiWinner != 0) {
            // ИИ должен быть ОЧЕНЬ уверен, а голосов за SAFE почти не должно быть
            if (aiWinnerWeight > 3.2 && aiSafeWeight < 0.8) {
                return new ArbitrationResult(aiWinner, "ОТКРЫТИЕ");
            }
            return new ArbitrationResult(0, "ОТСЕВ ШУМА (0)");
        }

        // 4. КОНСЕНСУС: Оба согласны на 0
        if (sonarClass == 0 && aiWinner == 0) {
            return new ArbitrationResult(0, "КОНСЕНСУС (0)");
        }

        // 5. ЖЕСТКИЙ КОНФЛИКТ
        // Если Sonar нашел одно, ИИ - другое, и они не в Топ-2 друг друга.
        return new ArbitrationResult(-1, "ОТБРАКОВКА (КОНФЛИКТ)");
    }
}