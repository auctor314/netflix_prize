use netflix_prize::{fit2, knn3::{Knn3Config, Knn3Model}, SPLIT_NEW};

fn main() {
    fit2!(Knn3Model, Knn3Config::default(), "1.0*mf-60", "mf-60__knn3", SPLIT_NEW);
}
