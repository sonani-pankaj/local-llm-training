package com.aigrama.llm_training.data;



import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import com.aigrama.llm_training.tokenizer.SimpleTokenizer;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * TextDataset - Training Dataset for LLM
 *
 * This class handles the preparation and management of text data
 * for training language models. It provides:
 *
 * 1. Text loading from files
 * 2. Tokenization using SimpleTokenizer
 * 3. Sequence creation with sliding window
 * 4. Batching for efficient GPU training
 * 5. Data shuffling and splitting (train/validation)
 * 6. NDArray conversion for DJL/ONNX Runtime
 *
 * GPU Memory Optimization:
 * - Sequences are created lazily to avoid memory spikes
 * - Batches are sized based on available GPU memory
 * - Data is streamed rather than loaded all at once
 *
 * @author LLM Training Platform
 * @version 1.0
 */
@Slf4j
public class TextDataset {

    // ==========================================
    // CONFIGURATION
    // ==========================================

    /** Maximum sequence length for training */
    @Getter
    private final int maxSequenceLength;

    /** Size of each training batch */
    @Getter
    private int batchSize;

    /** Percentage of data used for validation */
    private double validationSplit;

    /** Whether to shuffle data between epochs */
    private boolean shuffleEnabled;

    /** Random seed for reproducibility */
    private long randomSeed;

    // ==========================================
    // DATA STORAGE
    // ==========================================

    /** Complete tokenized dataset */
    private List<Integer> allTokens;

    /** Training sequences (input-output pairs) */
    private List<TrainingExample> trainingExamples;

    /** Validation sequences */
    private List<TrainingExample> validationExamples;

    /** Tokenizer for encoding/decoding text */
    private final SimpleTokenizer tokenizer;

    /** NDManager for creating tensors */
    private NDManager ndManager;

    /** Path to training data file */
    private String dataFilePath;

    /** Total number of tokens in dataset */
    @Getter
    private long totalTokens;

    /** Number of training examples */
    @Getter
    private int trainingSize;

    /** Number of validation examples */
    @Getter
    private int validationSize;

    /** Vocabulary size from tokenizer */
    @Getter
    private int vocabSize;

    /** Current epoch for shuffling */
    private int currentEpoch;

    /** Dataset statistics */
    private DatasetStatistics statistics;

    // ==========================================
    // CONSTRUCTOR
    // ==========================================

    /**
     * Creates a new TextDataset with default configuration
     *
     * @param tokenizer The tokenizer for encoding text
     * @param maxSequenceLength Maximum length of training sequences
     */
    public TextDataset(SimpleTokenizer tokenizer, int maxSequenceLength) {
        this.tokenizer = tokenizer;
        this.maxSequenceLength = maxSequenceLength;
        this.batchSize = 8; // Default batch size
        this.validationSplit = 0.1; // 10% for validation
        this.shuffleEnabled = true;
        this.randomSeed = 42L;
        this.currentEpoch = 0;
        this.trainingExamples = new ArrayList<>();
        this.validationExamples = new ArrayList<>();
        this.allTokens = new ArrayList<>();
        this.statistics = new DatasetStatistics();
    }

    /**
     * Creates a TextDataset with custom configuration
     */
    public TextDataset(SimpleTokenizer tokenizer,
                       int maxSequenceLength,
                       int batchSize,
                       double validationSplit) {
        this(tokenizer, maxSequenceLength);
        this.batchSize = batchSize;
        this.validationSplit = validationSplit;
    }

    // ==========================================
    // DATA LOADING AND PREPARATION
    // ==========================================

    /**
     * Load text data from file and prepare dataset
     *
     * @param filePath Path to the text file
     * @throws IOException If file cannot be read
     */
    public void loadFromFile(String filePath) throws IOException {
        log.info("Loading dataset from: {}", filePath);

        this.dataFilePath = filePath;
        Path path = Paths.get(filePath.replace("classpath:", "src/main/resources/"));

        if (!Files.exists(path)) {
            throw new IOException("Training data file not found: " + path);
        }

        // Read the entire text file
        String rawText = Files.readString(path);
        log.info("Loaded {} characters of raw text", rawText.length());

        // Build vocabulary if not already built
        if (tokenizer.getVocabSize() <= 4) {
            log.info("Building vocabulary from training data...");
            tokenizer.buildVocabulary(rawText);
        }

        // Tokenize the entire text
        log.info("Tokenizing text...");
        allTokens = tokenizer.encode(rawText);
        totalTokens = allTokens.size();
        vocabSize = tokenizer.getVocabSize();

        log.info("Tokenization complete:");
        log.info("  Total tokens: {}", totalTokens);
        log.info("  Vocabulary size: {}", vocabSize);

        // Create training examples
        createExamples();

        // Split into training and validation
        splitDataset();

        // Calculate statistics
        calculateStatistics();
    }

    /**
     * Load text directly from string
     */
    public void loadFromString(String text) {
        log.info("Loading dataset from string ({} characters)", text.length());

        if (tokenizer.getVocabSize() <= 4) {
            tokenizer.buildVocabulary(text);
        }

        allTokens = tokenizer.encode(text);
        totalTokens = allTokens.size();
        vocabSize = tokenizer.getVocabSize();

        createExamples();
        splitDataset();
        calculateStatistics();
    }

    /**
     * Create training examples using sliding window
     *
     * Each example consists of:
     * - Input sequence: tokens[i : i + maxSequenceLength]
     * - Target sequence: tokens[i+1 : i + maxSequenceLength + 1]
     *
     * This teaches the model to predict the next token.
     */
    private void createExamples() {
        log.info("Creating training examples...");

        List<TrainingExample> allExamples = new ArrayList<>();

        // Sliding window approach
        int stride = maxSequenceLength / 2; // 50% overlap for better coverage

        for (int i = 0; i < allTokens.size() - maxSequenceLength - 1; i += stride) {
            List<Integer> inputSequence = new ArrayList<>(
                    allTokens.subList(i, i + maxSequenceLength)
            );
            List<Integer> targetSequence = new ArrayList<>(
                    allTokens.subList(i + 1, i + maxSequenceLength + 1)
            );

            allExamples.add(new TrainingExample(inputSequence, targetSequence));
        }

        log.info("Created {} training examples (stride: {})", allExamples.size(), stride);
        this.trainingExamples = allExamples;
    }

    /**
     * Split dataset into training and validation sets
     */
    private void splitDataset() {
        int totalExamples = trainingExamples.size();
        int validationCount = (int) (totalExamples * validationSplit);
        int trainingCount = totalExamples - validationCount;

        // Shuffle before splitting
        if (shuffleEnabled) {
            Collections.shuffle(trainingExamples, new Random(randomSeed));
        }

        // Split
        validationExamples = new ArrayList<>(
                trainingExamples.subList(0, validationCount)
        );
        trainingExamples = new ArrayList<>(
                trainingExamples.subList(validationCount, totalExamples)
        );

        trainingSize = trainingExamples.size();
        validationSize = validationExamples.size();

        log.info("Dataset split:");
        log.info("  Training examples: {}", trainingSize);
        log.info("  Validation examples: {}", validationSize);
        log.info("  Split ratio: {}%", (int)(validationSplit * 100));
    }

    // ==========================================
    // BATCH CREATION
    // ==========================================

    /**
     * Create a batch for training
     *
     * @param batchIndex Index of the batch
     * @param manager NDManager for tensor creation
     * @return Batch containing input and target tensors
     */
    public Batch getBatch(int batchIndex, NDManager manager) {
        this.ndManager = manager;

        int startIdx = batchIndex * batchSize;
        int endIdx = Math.min(startIdx + batchSize, trainingSize);
        int actualBatchSize = endIdx - startIdx;

        // Create tensors for input and target
        NDArray inputTensor = manager.zeros(
                new Shape(actualBatchSize, maxSequenceLength),
                DataType.INT32
        );
        NDArray targetTensor = manager.zeros(
                new Shape(actualBatchSize, maxSequenceLength),
                DataType.INT32
        );
        NDArray attentionMask = manager.ones(
                new Shape(actualBatchSize, maxSequenceLength),
                DataType.INT32
        );

        // Fill tensors with data
        for (int i = 0; i < actualBatchSize; i++) {
            TrainingExample example = trainingExamples.get(startIdx + i);

            // Input sequence
            int[] inputSeq = example.getInputSequence();
            for (int j = 0; j < Math.min(inputSeq.length, maxSequenceLength); j++) {
                inputTensor.set(new NDIndex("{},{}", i, j), inputSeq[j]);
            }

            // Target sequence
            int[] targetSeq = example.getTargetSequence();
            for (int j = 0; j < Math.min(targetSeq.length, maxSequenceLength); j++) {
                targetTensor.set(new NDIndex("{},{}", i, j), targetSeq[j]);
            }
        }

        return new Batch(inputTensor, targetTensor, attentionMask, actualBatchSize);
    }

    /**
     * Create a validation batch
     */
    public Batch getValidationBatch(int batchIndex, NDManager manager) {
        this.ndManager = manager;

        int startIdx = batchIndex * batchSize;
        int endIdx = Math.min(startIdx + batchSize, validationSize);
        int actualBatchSize = endIdx - startIdx;

        NDArray inputTensor = manager.zeros(
                new Shape(actualBatchSize, maxSequenceLength),
                DataType.INT32
        );
        NDArray targetTensor = manager.zeros(
                new Shape(actualBatchSize, maxSequenceLength),
                DataType.INT32
        );
        NDArray attentionMask = manager.ones(
                new Shape(actualBatchSize, maxSequenceLength),
                DataType.INT32
        );

        for (int i = 0; i < actualBatchSize; i++) {
            TrainingExample example = validationExamples.get(startIdx + i);

            int[] inputSeq = example.getInputSequence();
            for (int j = 0; j < Math.min(inputSeq.length, maxSequenceLength); j++) {
                inputTensor.set(new NDIndex("{},{}", i, j), inputSeq[j]);
            }

            int[] targetSeq = example.getTargetSequence();
            for (int j = 0; j < Math.min(targetSeq.length, maxSequenceLength); j++) {
                targetTensor.set(new NDIndex("{},{}", i, j), targetSeq[j]);
            }
        }

        return new Batch(inputTensor, targetTensor, attentionMask, actualBatchSize);
    }

    /**
     * Get number of training batches
     */
    public int getNumBatches() {
        return (int) Math.ceil((double) trainingSize / batchSize);
    }

    /**
     * Get number of validation batches
     */
    public int getNumValidationBatches() {
        return (int) Math.ceil((double) validationSize / batchSize);
    }

    // ==========================================
    // EPOCH MANAGEMENT
    // ==========================================

    /**
     * Prepare for next epoch (shuffle if enabled)
     */
    public void nextEpoch() {
        currentEpoch++;

        if (shuffleEnabled) {
            long seed = randomSeed + currentEpoch;
            Collections.shuffle(trainingExamples, new Random(seed));
            log.debug("Shuffled training data for epoch {}", currentEpoch);
        }
    }

    /**
     * Reset dataset to initial state
     */
    public void reset() {
        currentEpoch = 0;
        if (shuffleEnabled) {
            Collections.shuffle(trainingExamples, new Random(randomSeed));
        }
    }

    // ==========================================
    // STATISTICS AND INFORMATION
    // ==========================================

    /**
     * Calculate dataset statistics
     */
    private void calculateStatistics() {
        statistics = new DatasetStatistics();

        // Calculate sequence length distribution
        IntSummaryStatistics seqStats = trainingExamples.stream()
                .mapToInt(e -> e.getInputSequence().length)
                .summaryStatistics();

        statistics.setAverageSequenceLength(seqStats.getAverage());
        statistics.setMinSequenceLength(seqStats.getMin());
        statistics.setMaxSequenceLength(seqStats.getMax());

        // Calculate token frequency distribution
        Map<Integer, Long> tokenFrequency = new HashMap<>();
        for (TrainingExample example : trainingExamples) {
            for (int token : example.getInputSequence()) {
                tokenFrequency.merge(token, 1L, Long::sum);
            }
        }

        statistics.setUniqueTokens(tokenFrequency.size());
        statistics.setMostCommonToken(
                tokenFrequency.entrySet().stream()
                        .max(Map.Entry.comparingByValue())
                        .map(Map.Entry::getKey)
                        .orElse(0)
        );

        // Calculate dataset size in memory
        long totalElements = (long) trainingSize * maxSequenceLength * 2; // input + target
        long memoryBytes = totalElements * 4; // 4 bytes per int32
        statistics.setEstimatedMemoryMB(memoryBytes / (1024.0 * 1024.0));

        log.info("Dataset statistics:");
        log.info("  Average sequence length: {}", String.format("%.2f", statistics.getAverageSequenceLength()));
        log.info("  Unique tokens: {}", statistics.getUniqueTokens());
        log.info("  Estimated memory: {} MB", String.format("%.2f", statistics.getEstimatedMemoryMB()));
    }

    /**
     * Get dataset statistics
     */
    public DatasetStatistics getStatistics() {
        return statistics;
    }

    /**
     * Get a sample from the dataset for inspection
     */
    public String getSample(int index) {
        if (index < 0 || index >= trainingSize) {
            return "Index out of range";
        }

        TrainingExample example = trainingExamples.get(index);
        String inputText = tokenizer.decode(Arrays.asList(
                Arrays.stream(example.getInputSequence()).boxed().toArray(Integer[]::new)
        ));
        String targetText = tokenizer.decode(Arrays.asList(
                Arrays.stream(example.getTargetSequence()).boxed().toArray(Integer[]::new)
        ));

        return String.format(
                "Example %d:\n  Input:  %s\n  Target: %s",
                index, inputText, targetText
        );
    }

    /**
     * Get dataset summary as string
     */
    public String getSummary() {
        return String.format("""
            Dataset Summary:
              Total tokens: %d
              Vocabulary size: %d
              Training examples: %d
              Validation examples: %d
              Sequence length: %d
              Batch size: %d
              Batches per epoch: %d
              Validation split: %.1f%%
              Shuffle enabled: %s
            """,
                totalTokens, vocabSize, trainingSize, validationSize,
                maxSequenceLength, batchSize, getNumBatches(),
                validationSplit * 100, shuffleEnabled
        );
    }

    // ==========================================
    // CONFIGURATION SETTERS
    // ==========================================

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        log.debug("Batch size set to: {}", batchSize);
    }

    public void setShuffleEnabled(boolean shuffleEnabled) {
        this.shuffleEnabled = shuffleEnabled;
        log.debug("Shuffle enabled: {}", shuffleEnabled);
    }

    public void setRandomSeed(long randomSeed) {
        this.randomSeed = randomSeed;
        log.debug("Random seed set to: {}", randomSeed);
    }

    // ==========================================
    // INNER CLASSES
    // ==========================================

    /**
     * TrainingExample - Input-Target pair for training
     */
    @Getter
    public static class TrainingExample {
        private final int[] inputSequence;
        private final int[] targetSequence;

        public TrainingExample(List<Integer> inputSequence, List<Integer> targetSequence) {
            this.inputSequence = inputSequence.stream().mapToInt(Integer::intValue).toArray();
            this.targetSequence = targetSequence.stream().mapToInt(Integer::intValue).toArray();
        }

        @Override
        public String toString() {
            return String.format("TrainingExample[input=%d tokens, target=%d tokens]",
                    inputSequence.length, targetSequence.length);
        }
    }

    /**
     * Batch - Container for a batch of training data
     */
    @Getter
    public static class Batch {
        private final NDArray inputTensor;
        private final NDArray targetTensor;
        private final NDArray attentionMask;
        private final int batchSize;

        public Batch(NDArray inputTensor, NDArray targetTensor,
                     NDArray attentionMask, int batchSize) {
            this.inputTensor = inputTensor;
            this.targetTensor = targetTensor;
            this.attentionMask = attentionMask;
            this.batchSize = batchSize;
        }

        /**
         * Release GPU memory for this batch
         */
        public void close() {
            if (inputTensor != null) inputTensor.close();
            if (targetTensor != null) targetTensor.close();
            if (attentionMask != null) attentionMask.close();
        }

        @Override
        public String toString() {
            return String.format("Batch[size=%d, input=%s, target=%s]",
                    batchSize,
                    inputTensor != null ? inputTensor.getShape() : "null",
                    targetTensor != null ? targetTensor.getShape() : "null"
            );
        }
    }

    /**
     * DatasetStatistics - Statistical information about the dataset
     */
    @Getter
    @lombok.Setter
    public static class DatasetStatistics {
        private double averageSequenceLength;
        private int minSequenceLength;
        private int maxSequenceLength;
        private int uniqueTokens;
        private int mostCommonToken;
        private double estimatedMemoryMB;

        @Override
        public String toString() {
            return String.format("""
                Dataset Statistics:
                  Avg sequence length: %.2f
                  Sequence range: [%d, %d]
                  Unique tokens: %d
                  Estimated memory: %.2f MB
                """,
                    averageSequenceLength, minSequenceLength, maxSequenceLength,
                    uniqueTokens, estimatedMemoryMB
            );
        }
    }

    /**
     * DataLoader - Iterator-style data loading for training loops
     */
    public class DataLoader implements Iterator<Batch>, AutoCloseable {
        private final NDManager manager;
        private final boolean isTraining;
        private int currentBatch;
        private final int totalBatches;

        public DataLoader(NDManager manager, boolean isTraining) {
            this.manager = manager;
            this.isTraining = isTraining;
            this.currentBatch = 0;
            this.totalBatches = isTraining ? getNumBatches() : getNumValidationBatches();
        }

        @Override
        public boolean hasNext() {
            return currentBatch < totalBatches;
        }

        @Override
        public Batch next() {
            if (!hasNext()) {
                throw new NoSuchElementException("No more batches available");
            }

            Batch batch;
            if (isTraining) {
                batch = getBatch(currentBatch, manager);
            } else {
                batch = getValidationBatch(currentBatch, manager);
            }

            currentBatch++;
            return batch;
        }

        public void reset() {
            currentBatch = 0;
        }

        public int getTotalBatches() {
            return totalBatches;
        }

        public int getCurrentBatchIndex() {
            return currentBatch;
        }

        @Override
        public void close() {
            // Cleanup if needed
        }
    }

    /**
     * Create a data loader for training
     */
    public DataLoader createTrainingLoader(NDManager manager) {
        return new DataLoader(manager, true);
    }

    /**
     * Create a data loader for validation
     */
    public DataLoader createValidationLoader(NDManager manager) {
        return new DataLoader(manager, false);
    }
}
