package org.example.parser.model;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class LLMVerdict {
    private int predictedClass; // 0-7 или -1 (если голова не идентифицировала уязвимость из 0-7 множества)
    private int confidence;     // уверенность в пересчёте 0-100
    private String headName;    // Какая именно голова дала ответ (H1, H2...)
}