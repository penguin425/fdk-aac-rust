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
    if !is_real_directory(&checkout) || !is_real_directory(&checkout.join(".git")) {
        if fs::symlink_metadata(&checkout).is_ok() {
            remove_checkout(&checkout);
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
    if !command_succeeds(Command::new("git").args(["-C"]).arg(&checkout).args([
        "cat-file",
        "-e",
        &format!("{revision}^{{commit}}"),
    ])) {
        run(Command::new("git").args(["-C"]).arg(&checkout).args([
            "fetch",
            "--quiet",
            "--depth=1",
            "origin",
            &revision,
        ]));
    }
    run(Command::new("git")
        .args(["-C"])
        .arg(&checkout)
        .args(["reset", "--hard", "--quiet", &revision]));
    run(Command::new("git")
        .args(["-C"])
        .arg(&checkout)
        .args(["clean", "-fdx", "--quiet"]));
    let actual_revision = run_output(
        Command::new("git")
            .args(["-C"])
            .arg(&checkout)
            .args(["rev-parse", "HEAD"]),
    );
    assert_eq!(
        actual_revision, revision,
        "upstream checkout revision mismatch"
    );
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

fn is_real_directory(path: &std::path::Path) -> bool {
    fs::symlink_metadata(path)
        .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
}

fn remove_checkout(path: &std::path::Path) {
    let metadata = fs::symlink_metadata(path).expect("inspect incomplete upstream checkout");
    if metadata.is_dir() && !metadata.file_type().is_symlink() {
        fs::remove_dir_all(path).expect("remove incomplete upstream checkout directory");
    } else {
        fs::remove_file(path).expect("remove incomplete upstream checkout entry");
    }
}

fn run(command: &mut Command) {
    let display = format!("{command:?}");
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("failed to run {display}: {error}"));
    assert!(status.success(), "command failed: {display}");
}

fn command_succeeds(command: &mut Command) -> bool {
    command.status().is_ok_and(|status| status.success())
}

fn run_output(command: &mut Command) -> String {
    let display = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {display}: {error}"));
    assert!(output.status.success(), "command failed: {display}");
    String::from_utf8(output.stdout)
        .expect("command output is not UTF-8")
        .trim()
        .to_owned()
}
