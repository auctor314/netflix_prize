use netflix_prize::{fit2, mf::{MfConfig, MfModel}, SPLIT_NEW};

fn main() {
    let cfg = MfConfig {
        n_feat: 60,
        n_epochs: 12,
        seed: 42,
        shuffle_users: true,
        lr_u: 0.0031,
        lr_i: 0.0036,
        lr_ub: 0.0031,
        lr_ib: 0.0036,
        reg_u: 0.03,
        reg_i: 0.005,
        sigma_u: 0.004,
        sigma_i: 0.005,
        reset_u_epoch: 10,
        item_feat_npy: None,
        ordinal_head: None,
        save_ifeat: false,
    };
    fit2!(MfModel, cfg, "rtg", "mf-60", SPLIT_NEW, save_train: true);
}
