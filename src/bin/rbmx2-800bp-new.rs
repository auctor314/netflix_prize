use netflix_prize::{fit2, rx::{HiddenType, RxConfig, RxModel, VisibleType}, SPLIT_NEW};

fn main() {
    let cfg = RxConfig {
        hidden_type: HiddenType::Bipolar,
        visible_type: VisibleType::Softmax,
        temperature: 1.0,
        n_hidden: 800,
        n_epochs: 3,
        seed: 700,
        shuffle_users: true,
        init_sigma: 0.01,
        batch_size: 1000,
        lr: 0.02,
        momentum: 0.9,
        weight_decay: 0.001,
        lr_bu: 0.001,
        wd_bu: 0.01,
        lr_but: 0.0005,
        wd_but: 0.01,
        cd_start: 1,
        cd_inc_every: 5,
        cd_inc_by: 1,
        cd_max: 5,
        use_conditional: true,
        r_include_pr_all: true,
        save_w: false,
        n_factors: None,
        n_freq_bins: 8,
        lr_bif: 0.0,
        wd_bif: 0.01,
    };
    fit2!(
		RxModel, cfg, "rtg", "rbmx2-800bp", SPLIT_NEW,
		save_probe_each_epoch: true
	);
}
