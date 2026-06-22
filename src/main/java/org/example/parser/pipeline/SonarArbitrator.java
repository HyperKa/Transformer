package org.example.parser.pipeline;

import org.example.parser.model.SonarVerdict;
import org.example.parser.pipeline.MistralArbitrator.AIResult;
import java.util.*;

public class SonarArbitrator {

    public static class ArbitrationResult {
        public final int label;
        public final String status;

        public ArbitrationResult(int label, String status) {
            this.label = label;
            this.status = status;
        }
    }

    // Иерархия критичности угроз
    private static final List<Integer> SEVERITY_ORDER = List.of(5, 6, 3, 7, 1, 4, 2, 0);

    public static ArbitrationResult determineFinalLabel(AIResult ai, SonarVerdict sonar) {
        int sonarClass = sonar.getPredictedClass();
        int aiWinner = ai.getWinner();
        double aiWinnerWeight = ai.getWeight(aiWinner);
        int architectVerdict = ai.getSpecificHeadVerdict("H17_ARCHITECT");

        // ПРАВИЛО 1: ВЕТО АРХИТЕКТОРА (Если архитектор уверен в SAFE)
        if (architectVerdict == 0 && ai.getWeight(0) > 2.2) {
            return new ArbitrationResult(0, "ВЕТО_АРХИТЕКТОРА (SAFE)");
        }

        // ПРАВИЛО 2: ЕДИНОЛИЧНЫЙ БЕЗОГОВОРОЧНЫЙ ЛИДЕР
        // Если победитель опережает любого другого кандидата минимум на 1.0 балл,
        // мы отдаем победу ему, игнорируя иерархию (исключаем подавление шумом)
        boolean isUncontestedLeader = true;
        for (int i = 0; i <= 7; i++) {
            if (i != aiWinner && (aiWinnerWeight - ai.getWeight(i)) < 1.0) {
                isUncontestedLeader = false; // Есть близкий соперник, это реальный спор
                break;
            }
        }

        if (isUncontestedLeader && aiWinnerWeight > 1.5) {
            String status = (aiWinner == sonarClass) ? "СИНЕРГИЯ" : "ЛИДЕР_КОМИТЕТА";
            return new ArbitrationResult(aiWinner, status);
        }

        // ПРАВИЛО 3: СПОРНЫЙ КЕЙС (Разница между лидерами < 1.0 балла)
        // Включаем иерархию приоритетов только среди РЕАЛЬНО конкурирующих классов
        Set<Integer> suspects = new java.util.HashSet<>();
        if (sonarClass != 0) {
            suspects.add(sonarClass);
        }

        for (int i = 0; i <= 7; i++) {
            // Добавляем в спор только сильных кандидатов, близких к весу победителя
            if (ai.getWeight(i) > 1.2 && (aiWinnerWeight - ai.getWeight(i)) < 1.0) {
                suspects.add(i);
            }
        }

        // Выбираем по иерархии критичности среди спорных альтернатив
        for (Integer targetClass : SEVERITY_ORDER) {
            if (suspects.contains(targetClass)) {
                String status = "ИЕРАРХИЯ_КОНФЛИКТА";
                if (targetClass == sonarClass) {
                    status = "СИНЕРГИЯ_ИЕРАРХИИ";
                }
                return new ArbitrationResult(targetClass, status);
            }
        }

        // Дефолтный выход
        return new ArbitrationResult(aiWinner, "ДЕФОЛТ_ПОБЕДИТЕЛЬ");
    }
}