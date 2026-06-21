package com.aigrama.llm_training.tokenizer;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@Slf4j
public class SimpleTokenizer {

    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;
    @Getter
    private final int maxVocabSize;

    // Java 25: Using String templates for cleaner logging
    public SimpleTokenizer(int maxVocabSize) {
        this.maxVocabSize = maxVocabSize;
        this.tokenToId = new ConcurrentHashMap<>();
        this.idToToken = new ConcurrentHashMap<>();

        // Add special tokens using immutable collections
        addSpecialToken("<PAD>"); // 0 - Padding
        addSpecialToken("<UNK>"); // 1 - Unknown
        addSpecialToken("<BOS>"); // 2 - Beginning of Sequence
        addSpecialToken("<EOS>"); // 3 - End of Sequence

        log.info(String.format("Initialized tokenizer with max vocab size: %d", maxVocabSize));
    }

    private void addSpecialToken(String token) {
        int id = tokenToId.size();
        tokenToId.put(token, id);
        idToToken.put(id, token);
    }

    /**
     * Build vocabulary from training text using frequency-based selection
     */
    public void buildVocabulary(String text) {
        log.info("Building vocabulary from text of length: {}", text.length());

        // Java 25: Using stream improvements and pattern matching
        var words = text.toLowerCase()
                .replaceAll("[^a-zA-Z0-9\\s]", " ")
                .lines()
                .flatMap(line -> Arrays.stream(line.split("\\s+")))
                .filter(word -> !word.isEmpty())
                .toList();

        // Count word frequencies using modern collectors
        var wordCount = words.stream()
                .collect(Collectors.groupingBy(
                        word -> word,
                        Collectors.counting()
                ));

        // Select top words by frequency, reserving space for special tokens
        int availableSlots = maxVocabSize - tokenToId.size();

        wordCount.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(availableSlots)
                .forEach(entry -> {
                    int id = tokenToId.size();
                    tokenToId.put(entry.getKey(), id);
                    idToToken.put(id, entry.getKey());
                });

        log.info(String.format("Built vocabulary with %d tokens", tokenToId.size()));
    }

    /**
     * Encode text to token IDs
     */
    public List<Integer> encode(String text) {
        Objects.requireNonNull(text, "Text cannot be null");

        var tokens = new ArrayList<Integer>();
        tokens.add(tokenToId.get("<BOS>"));

        // Java 25: Using text blocks for regex patterns
        var pattern = """
            [^a-zA-Z0-9\\s]
            """;

        var words = text.toLowerCase()
                .replaceAll(pattern.strip(), " ")
                .split("\\s+");

        for (var word : words) {
            if (!word.isEmpty()) {
                tokens.add(tokenToId.getOrDefault(word, tokenToId.get("<UNK>")));
            }
        }

        tokens.add(tokenToId.get("<EOS>"));
        return Collections.unmodifiableList(tokens);
    }

    /**
     * Decode token IDs back to text
     */
    public String decode(List<Integer> tokenIds) {
        // Java 25: Using collectors with teeing and improved joining
        return tokenIds.stream()
                .map(id -> idToToken.getOrDefault(id, "<UNK>"))
                .filter(token -> !Set.of("<PAD>", "<BOS>", "<EOS>").contains(token))
                .collect(Collectors.joining(" "));
    }

    public int getVocabSize() {
        return tokenToId.size();
    }

    public int getPadId() {
        return 0;
    }

    public int getEosId() {
        return 3;
    }
}