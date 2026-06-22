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

    // Табель о рангах (от критических уязвимостей к менее опасным)
    private static final List<Integer> SEVERITY_ORDER = List.of(5, 6, 3, 7, 1, 4, 2, 0);

    public static ArbitrationResult determineFinalLabel(AIResult ai, SonarVerdict sonar) {
        int sonarClass = sonar.getPredictedClass();
        int aiWinner = ai.getWinner();
        int architectVerdict = ai.getSpecificHeadVerdict("H17_ARCHITECT");

        // Сбор всех подтвержденных подозрений (вес класса > 1.2 или совпадение с линтером)
        Set<Integer> activeSuspects = new java.util.HashSet<>();
        if (sonarClass != 0) {
            activeSuspects.add(sonarClass);
        }

        for (int i = 1; i <= 7; i++) {
            if (ai.getWeight(i) >= 1.2) {
                activeSuspects.add(i);
            }
        }

        // Если есть вето от архиватора на ложное срабатывание уязвимости
        if (architectVerdict == 0 && ai.getWeight(0) > 2.5) {
            return new ArbitrationResult(0, "ВЕТО_АРХИТЕКТОРА (SAFE)");
        }

        // Выбираем самую опасную уязвимость из подтвержденных подозрений
        for (Integer targetClass : SEVERITY_ORDER) {
            if (activeSuspects.contains(targetClass)) {
                String status = "МНОГОКЛАССОВЫЙ_КОНФЕНСУС";
                if (targetClass == sonarClass) {
                    status = "СИНЕРГИЯ_17_ГОЛОВ";
                } else if (targetClass == aiWinner) {
                    status = "РЕШЕНИЕ_КОМИТЕТА";
                } else if (targetClass == architectVerdict) {
                    status = "РЕШЕНИЕ_АРХИТЕКТОРА";
                }
                return new ArbitrationResult(targetClass, "ИЕРАРХИЯ:" + status);
            }
        }

        // Если никто ничего не нашел - код безопасен
        return new ArbitrationResult(0, "АБСОЛЮТНЫЙ_SAFE");
    }
}