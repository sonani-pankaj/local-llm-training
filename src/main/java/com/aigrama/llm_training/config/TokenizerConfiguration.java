package com.aigrama.llm_training.config;


import com.aigrama.llm_training.tokenizer.SimpleTokenizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class TokenizerConfiguration {

    private final TrainingConfiguration config;

    public TokenizerConfiguration(TrainingConfiguration config) {
        this.config = config;
    }

    @Bean
    public SimpleTokenizer tokenizer() {
        return new SimpleTokenizer(config.getModel().getVocabSize());
    }
}