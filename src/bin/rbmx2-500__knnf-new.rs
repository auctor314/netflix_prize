use netflix_prize::{fit2, knnf::{KnnfConfig, KnnfModel}, SPLIT_NEW};

fn main() {
    let cfg = KnnfConfig {
        factors: "preds_new/rbmx2-500.ifeat",
        k: 15,
        scaling: 1.5,
        tau: 0.0,
    };
    fit2!(KnnfModel, cfg, "1.1*rbmx2-500", "rbmx2-500__knnf", SPLIT_NEW);
}
