use kube::CustomResourceExt;

fn main() {
    let crd = discovery::crd::PodCache::crd();
    print!(
        "{}",
        serde_yaml::to_string(&crd).expect("failed to serialize CRD")
    );
}
