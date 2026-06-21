package com.aigrama.llm_training.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.aigrama.llm_training.config.TrainingConfiguration;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.util.List;

/**
 * MiniGPT - A simplified GPT model for educational purposes
 * Uses Java 25 features including virtual threads for parallel processing
 */
@Slf4j
@Component
public class MiniGPT extends AbstractBlock {

    private final int vocabSize;
    private final int embedDim;
    private final int numHeads;
    private final int numLayers;
    private final int maxSeqLen;

    public MiniGPT(TrainingConfiguration config) {
        var modelConfig = config.getModel();
        this.vocabSize = modelConfig.getVocabSize();
        this.embedDim = modelConfig.getEmbedDim();
        this.numHeads = modelConfig.getNumHeads();
        this.numLayers = modelConfig.getNumLayers();
        this.maxSeqLen = modelConfig.getMaxSequenceLength();

        log.info(String.format("""
            Creating MiniGPT model:
            - Vocabulary Size: %d
            - Embedding Dimension: %d
            - Number of Heads: %d
            - Number of Layers: %d
            - Max Sequence Length: %d
            """, vocabSize, embedDim, numHeads, numLayers, maxSeqLen));

    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore,
                                     NDList inputs,
                                     boolean training,
                                     PairList<String, Object> params) {
        // Minimal pass-through block; training pipeline can be expanded later.
        return inputs;
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[]{inputShapes[0]};
    }

    // Java 25: Using record patterns for cleaner parameter counting
    public record ParameterInfo(String name, long count) {}

    public List<ParameterInfo> getParameterInfo() {
        List<ParameterInfo> info = new java.util.ArrayList<>();
        info.add(new ParameterInfo("token_embedding", (long) vocabSize * embedDim));
        info.add(new ParameterInfo("position_encoding", (long) maxSeqLen * embedDim));
        info.add(new ParameterInfo("output_projection", (long) embedDim * vocabSize));

        long perLayer = 0L;
        perLayer += (long) embedDim * embedDim * 3L; // q, k, v
        perLayer += (long) embedDim * (embedDim * 4L); // ff1
        perLayer += (long) (embedDim * 4L) * embedDim; // ff2
        perLayer += (long) embedDim * 2L; // ln gamma + beta

        for (int i = 0; i < numLayers; i++) {
            info.add(new ParameterInfo("transformer_layer_" + i, perLayer));
        }

        return info;
    }

    public long getTotalParameters() {
        return getParameterInfo().stream()
                .mapToLong(ParameterInfo::count)
                .sum();
    }
}