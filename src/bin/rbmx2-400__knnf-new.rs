use netflix_prize::{fit2, knnf::{KnnfConfig, KnnfModel}, SPLIT_NEW};

fn main() {
    fit2!(
        KnnfModel,
        KnnfConfig::with_factors("preds_new/rbmx2-400.ifeat"),
        "1.0*rbmx2-400",
        "rbmx2-400__knnf",
        SPLIT_NEW
    );
}
