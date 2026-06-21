package com.aigrama.llm_training.data;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Random;

/**
 * SampleDataGenerator - Generates training data for LLM
 * Creates diverse text samples covering multiple domains:
 * - Technology and AI
 * - Science and Mathematics
 * - Business and Economics
 * - History and Literature
 * - Programming and Development
 */
@Slf4j
@Component
public class SampleDataGenerator {

    private static final String[] SENTENCE_TEMPLATES = {
            // AI and Machine Learning
            "Artificial intelligence continues to revolutionize {domain} through innovative {technology} approaches.",
            "Deep learning models have achieved {adjective} performance on {task} benchmarks this year.",
            "The transformer architecture enables {benefit} through its innovative {mechanism} mechanism.",
            "Training {model_type} requires {resource} and careful {process} to achieve optimal results.",
            "Natural language processing has made {progress} progress in understanding {aspect} of human language.",

            // Technology and Computing
            "{language} programming language offers {feature} for building {application_type} applications.",
            "Cloud computing provides {benefit} through {service_type} services at {advantage} costs.",
            "Cybersecurity threats continue to {action} requiring {solution} to protect {target} systems.",
            "The Internet of Things connects {device_count} devices generating {data_amount} of data daily.",
            "Blockchain technology enables {benefit} through {mechanism} consensus mechanisms.",

            // Science and Research
            "Researchers have discovered {discovery} that could lead to {impact} in the field of {field}.",
            "The study of {subject} reveals {finding} about the nature of {phenomenon} in our universe.",
            "Scientific breakthroughs in {field} have enabled {achievement} through {method} approaches.",
            "Understanding {concept} is fundamental to advancing our knowledge of {broader_field}.",
            "The relationship between {factor_a} and {factor_b} demonstrates {principle} in action.",

            // Business and Economics
            "Market trends indicate {trend} in {sector} driven by {driver} and changing {factor}.",
            "Digital transformation is reshaping {industry} through {technology} and {approach} strategies.",
            "Sustainable business practices lead to {benefit} while addressing {challenge} concerns.",
            "The global economy faces {challenge} requiring {solution} from {stakeholder} worldwide.",
            "Innovation in {field} creates opportunities for {outcome} and sustainable {growth}.",

            // Programming and Development
            "Software architecture patterns like {pattern} help manage {concern} in {system_type} systems.",
            "Version control systems enable {benefit} through features like {feature} and {another_feature}.",
            "Testing methodologies including {method} ensure software quality through {approach} verification.",
            "Database optimization techniques such as {technique} improve {metric} by significant margins.",
            "API design principles emphasize {principle} to create {quality} interfaces for developers."
    };

    private static final String[] DOMAINS = {
            "healthcare", "finance", "education", "transportation", "manufacturing",
            "retail", "agriculture", "energy", "entertainment", "communication"
    };

    private static final String[] TECHNOLOGIES = {
            "machine learning", "deep learning", "computer vision", "natural language processing",
            "reinforcement learning", "generative AI", "robotics", "edge computing"
    };

    public static void generateSampleData(String outputPath, int numSentences) throws IOException {
        Random random = new Random(42);
        StringBuilder text = new StringBuilder();

        log.info("Generating {} sample sentences...", numSentences);

        for (int i = 0; i < numSentences; i++) {
            String template = SENTENCE_TEMPLATES[random.nextInt(SENTENCE_TEMPLATES.length)];
            String sentence = fillTemplate(template, random);
            text.append(sentence).append("\n");

            if ((i + 1) % 100 == 0) {
                log.info("Generated {} sentences...", i + 1);
            }
        }

        Path path = Paths.get(outputPath);
        Files.writeString(path, text.toString());

        log.info("Generated {} sentences saved to {}", numSentences, outputPath);
        log.info("Total characters: {}", text.length());
    }

    private static String fillTemplate(String template, Random random) {
        return template
                .replace("{domain}", getRandom(DOMAINS, random))
                .replace("{technology}", getRandom(TECHNOLOGIES, random))
                .replace("{adjective}", getRandom(ADJECTIVES, random))
                .replace("{benefit}", getRandom(BENEFITS, random))
                .replace("{mechanism}", getRandom(MECHANISMS, random))
                .replace("{task}", getRandom(TASKS, random))
                .replace("{model_type}", getRandom(MODEL_TYPES, random))
                .replace("{resource}", getRandom(RESOURCES, random))
                .replace("{process}", getRandom(PROCESSES, random))
                .replace("{progress}", getRandom(PROGRESS_LEVELS, random))
                .replace("{aspect}", getRandom(ASPECTS, random))
                .replace("{language}", getRandom(LANGUAGES, random))
                .replace("{feature}", getRandom(FEATURES, random))
                .replace("{application_type}", getRandom(APPLICATION_TYPES, random))
                .replace("{service_type}", getRandom(SERVICE_TYPES, random))
                .replace("{advantage}", getRandom(ADVANTAGES, random))
                .replace("{action}", getRandom(ACTIONS, random))
                .replace("{solution}", getRandom(SOLUTIONS, random))
                .replace("{target}", getRandom(TARGETS, random))
                .replace("{device_count}", String.valueOf(random.nextInt(50) + 10))
                .replace("{data_amount}", getRandom(DATA_AMOUNTS, random))
                .replace("{discovery}", getRandom(DISCOVERIES, random))
                .replace("{impact}", getRandom(IMPACTS, random))
                .replace("{field}", getRandom(FIELDS, random))
                .replace("{subject}", getRandom(SUBJECTS, random))
                .replace("{finding}", getRandom(FINDINGS, random))
                .replace("{phenomenon}", getRandom(PHENOMENA, random))
                .replace("{achievement}", getRandom(ACHIEVEMENTS, random))
                .replace("{method}", getRandom(METHODS, random))
                .replace("{concept}", getRandom(CONCEPTS, random))
                .replace("{broader_field}", getRandom(BROADER_FIELDS, random))
                .replace("{factor_a}", getRandom(FACTORS_A, random))
                .replace("{factor_b}", getRandom(FACTORS_B, random))
                .replace("{principle}", getRandom(PRINCIPLES, random))
                .replace("{trend}", getRandom(TRENDS, random))
                .replace("{sector}", getRandom(SECTORS, random))
                .replace("{driver}", getRandom(DRIVERS, random))
                .replace("{factor}", getRandom(FACTORS, random))
                .replace("{industry}", getRandom(INDUSTRIES, random))
                .replace("{approach}", getRandom(APPROACHES, random))
                .replace("{challenge}", getRandom(CHALLENGES, random))
                .replace("{stakeholder}", getRandom(STAKEHOLDERS, random))
                .replace("{outcome}", getRandom(OUTCOMES, random))
                .replace("{growth}", getRandom(GROWTH_TYPES, random))
                .replace("{pattern}", getRandom(PATTERNS, random))
                .replace("{concern}", getRandom(CONCERNS, random))
                .replace("{system_type}", getRandom(SYSTEM_TYPES, random))
                .replace("{another_feature}", getRandom(FEATURES, random))
                .replace("{technique}", getRandom(TECHNIQUES, random))
                .replace("{metric}", getRandom(METRICS, random))
                .replace("{quality}", getRandom(QUALITIES, random));
    }

    private static String getRandom(String[] array, Random random) {
        return array[random.nextInt(array.length)];
    }

    // Word banks for template filling
    private static final String[] ADJECTIVES = {"remarkable", "significant", "unprecedented", "substantial", "notable"};
    private static final String[] BENEFITS = {"scalability", "efficiency", "accuracy", "performance", "reliability"};
    private static final String[] MECHANISMS = {"attention", "self-attention", "cross-attention", "feedforward", "normalization"};
    private static final String[] TASKS = {"classification", "generation", "translation", "summarization", "question-answering"};
    private static final String[] MODEL_TYPES = {"transformer models", "language models", "neural networks", "deep learning models", "foundation models"};
    private static final String[] RESOURCES = {"substantial computational resources", "large datasets", "careful hyperparameter tuning", "extensive training time", "specialized hardware"};
    private static final String[] PROCESSES = {"optimization", "regularization", "fine-tuning", "pre-training", "validation"};
    private static final String[] PROGRESS_LEVELS = {"remarkable", "impressive", "significant", "notable", "substantial"};
    private static final String[] ASPECTS = {"semantic", "syntactic", "contextual", "pragmatic", "discourse"};
    private static final String[] LANGUAGES = {"Python", "Java", "JavaScript", "Rust", "Go"};
    private static final String[] FEATURES = {"asynchronous programming", "type safety", "memory management", "pattern matching", "functional programming"};
    private static final String[] APPLICATION_TYPES = {"scalable", "enterprise", "cloud-native", "real-time", "distributed"};
    private static final String[] SERVICE_TYPES = {"platform", "infrastructure", "software", "function", "database"};
    private static final String[] ADVANTAGES = {"reduced", "lower", "minimal", "competitive", "predictable"};
    private static final String[] ACTIONS = {"evolve rapidly", "become more sophisticated", "target critical infrastructure", "exploit vulnerabilities", "bypass defenses"};
    private static final String[] SOLUTIONS = {"advanced encryption", "multi-factor authentication", "zero-trust architectures", "continuous monitoring", "AI-powered detection"};
    private static final String[] TARGETS = {"enterprise", "government", "healthcare", "financial", "educational"};
    private static final String[] DATA_AMOUNTS = {"petabytes", "exabytes", "zettabytes", "terabytes", "massive amounts"};
    private static final String[] DISCOVERIES = {"a breakthrough treatment", "a novel material", "an important mechanism", "a fundamental principle", "an unexpected relationship"};
    private static final String[] IMPACTS = {"revolutionary changes", "significant improvements", "transformative applications", "widespread adoption", "paradigm shifts"};
    private static final String[] FIELDS = {"medicine", "physics", "biology", "chemistry", "materials science"};
    private static final String[] SUBJECTS = {"quantum mechanics", "molecular biology", "neuroscience", "climate science", "astrophysics"};
    private static final String[] FINDINGS = {"surprising patterns", "unexpected correlations", "fundamental principles", "novel mechanisms", "critical insights"};
    private static final String[] PHENOMENA = {"consciousness", "dark matter", "quantum entanglement", "emergence", "complexity"};
    private static final String[] ACHIEVEMENTS = {"unprecedented precision", "groundbreaking discoveries", "remarkable advances", "transformative capabilities", "revolutionary insights"};
    private static final String[] METHODS = {"computational", "experimental", "theoretical", "observational", "interdisciplinary"};
    private static final String[] CONCEPTS = {"entropy", "symmetry", "conservation", "emergence", "feedback"};
    private static final String[] BROADER_FIELDS = {"science", "engineering", "philosophy", "mathematics", "technology"};
    private static final String[] FACTORS_A = {"temperature", "pressure", "concentration", "velocity", "frequency"};
    private static final String[] FACTORS_B = {"reaction rate", "system behavior", "output quality", "performance", "efficiency"};
    private static final String[] PRINCIPLES = {"conservation laws", "thermodynamic principles", "quantum effects", "relativistic corrections", "fundamental symmetries"};
    private static final String[] TRENDS = {"accelerating adoption", "increasing investment", "growing demand", "rapid innovation", "market consolidation"};
    private static final String[] SECTORS = {"technology", "healthcare", "finance", "energy", "manufacturing"};
    private static final String[] DRIVERS = {"technological innovation", "regulatory changes", "consumer preferences", "competitive pressure", "environmental concerns"};
    private static final String[] FACTORS = {"market", "environmental", "regulatory", "technological", "social"};
    private static final String[] INDUSTRIES = {"manufacturing", "healthcare", "retail", "finance", "education"};
    private static final String[] APPROACHES = {"data-driven", "customer-centric", "agile", "sustainable", "innovative"};
    private static final String[] CHALLENGES = {"rising costs", "supply chain disruptions", "talent shortages", "regulatory complexity", "technological disruption"};
    private static final String[] STAKEHOLDERS = {"governments", "corporations", "communities", "institutions", "organizations"};
    private static final String[] OUTCOMES = {"economic growth", "social progress", "environmental sustainability", "technological advancement", "improved quality of life"};
    private static final String[] GROWTH_TYPES = {"economic", "sustainable", "inclusive", "technological", "organizational"};
    private static final String[] PATTERNS = {"MVC", "microservices", "event-driven", "CQRS", "repository"};
    private static final String[] CONCERNS = {"scalability", "maintainability", "security", "performance", "reliability"};
    private static final String[] SYSTEM_TYPES = {"distributed", "embedded", "real-time", "cloud-based", "enterprise"};
    private static final String[] TECHNIQUES = {"indexing", "caching", "sharding", "replication", "partitioning"};
    private static final String[] METRICS = {"query performance", "throughput", "latency", "resource utilization", "scalability"};
    private static final String[] QUALITIES = {"robust", "scalable", "maintainable", "secure", "performant"};
    private static final String[] SOLUTIONS_ARRAY = {"innovative solutions", "comprehensive strategies", "coordinated responses", "collaborative approaches", "systematic reforms"};
    private static final String[] TARGETS_ARRAY = {"sensitive", "critical", "vulnerable", "essential", "protected"};

    public static void main(String[] args) throws IOException {
        generateSampleData("src/main/resources/training_data.txt", 1000);
    }
}