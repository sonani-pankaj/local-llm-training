package com.aigrama.llm_training.data;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

@Slf4j
@Component
public class DataGenerator {

    public static void createSampleData() throws IOException {
        // Java 25: Using text blocks and string templates
        var templates = new String[]{
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "Machine learning models require large amounts of data.",
                "Neural networks are inspired by biological brains.",
                "Deep learning has achieved remarkable results in NLP tasks.",
                "Transformer architecture revolutionized language processing.",
                "Training large language models requires significant compute.",
                "Java is a versatile programming language for enterprise."
        };

        var text = new StringBuilder();
        for (int i = 0; i < 100; i++) {
            for (var template : templates) {
                text.append(template).append(" ");
            }
        }

        Path outputPath = Path.of("src", "main", "resources", "training_data.txt");
        Files.writeString(outputPath, text.toString());
        log.info("Generated training data: {} characters", text.length());
    }

    public static String loadTrainingData(String path) throws IOException {
        return Files.readString(Path.of(path.replace("classpath:", "src/main/resources/")));
    }
}
