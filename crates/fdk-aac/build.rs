use std::{env, fs, path::PathBuf, process::Command};

const UPSTREAM_URL: &str = "https://github.com/mstorsjo/fdk-aac.git";
const DEFAULT_UPSTREAM_REVISION: &str = "d8e6b1a3aa606c450241632b64b703f21ea31ce3";

fn main() {
    println!("cargo:rerun-if-env-changed=FDK_AAC_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=FDK_AAC_REVISION");
    let source = upstream_source();
    println!("cargo:rustc-env=FDK_AAC_UPSTREAM_DIR={}", source.display());
}

fn upstream_source() -> PathBuf {
    if let Some(path) = env::var_os("FDK_AAC_SOURCE_DIR") {
        let path = PathBuf::from(path);
        require_source_tree(&path);
        return path
            .canonicalize()
            .expect("canonicalize FDK_AAC_SOURCE_DIR");
    }

    let revision = upstream_revision();
    let checkout = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"))
        .join(format!("fdk-aac-upstream-{revision}"));
    if !checkout.join("CMakeLists.txt").is_file() {
        if checkout.exists() {
            fs::remove_dir_all(&checkout).expect("remove incomplete upstream checkout");
        }
        fs::create_dir_all(&checkout).expect("create upstream checkout directory");
        run(Command::new("git")
            .arg("init")
            .arg("--quiet")
            .arg(&checkout));
        run(Command::new("git").args(["-C"]).arg(&checkout).args([
            "remote",
            "add",
            "origin",
            UPSTREAM_URL,
        ]));
        run(Command::new("git").args(["-C"]).arg(&checkout).args([
            "fetch",
            "--quiet",
            "--depth=1",
            "origin",
            &revision,
        ]));
        run(Command::new("git").args(["-C"]).arg(&checkout).args([
            "checkout",
            "--quiet",
            "--detach",
            "FETCH_HEAD",
        ]));
    }
    require_source_tree(&checkout);
    checkout
}

fn upstream_revision() -> String {
    let revision =
        env::var("FDK_AAC_REVISION").unwrap_or_else(|_| DEFAULT_UPSTREAM_REVISION.to_owned());
    assert!(
        revision.len() == 40 && revision.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "FDK_AAC_REVISION must be a full 40-character Git commit SHA"
    );
    revision
}

fn require_source_tree(path: &std::path::Path) {
    if !path.join("CMakeLists.txt").is_file()
        || !path.join("libFDK/src/FDK_tools_rom.cpp").is_file()
    {
        panic!("{} is not an fdk-aac source tree", path.display());
    }
}

fn run(command: &mut Command) {
    let display = format!("{command:?}");
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {display}: {error}"));
    assert!(status.success(), "command failed: {display}");
}
