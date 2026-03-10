use kube::CustomResourceExt;

fn main() {
    print!(
        "{}",
        serde_yaml::to_string(&discovery::crd::NodeCache::crd()).unwrap()
    );
}
