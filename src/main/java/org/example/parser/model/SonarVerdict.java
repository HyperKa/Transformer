package org.example.parser.model;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class SonarVerdict {
    private int predictedClass; // 0-7
    private double confidence;  // 0.0 - 1.0 (у Sonar всегда 1.0, если нашел)
    private String ruleName;    // например, "java:S3649"
}