package org.example.parser.model;

import lombok.Data;
import java.time.LocalDateTime;

@Data
public class CodeCandidate {
    private String id;
    private String source;
    private String cwe;
    private Integer finalClass;
    private String methodType;
    private String fullContext;
    private String vulnerableMethod;
    private String filePath;
    private LocalDateTime createdAt;
    private int codeLength;
}
