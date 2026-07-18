use std::{env, fs, path::PathBuf, process::Command};

const UPSTREAM_URL: &str = "https://github.com/mstorsjo/fdk-aac.git";
const DEFAULT_UPSTREAM_REVISION: &str = "d8e6b1a3aa606c450241632b64b703f21ea31ce3";

fn main() {
    println!("cargo:rerun-if-env-changed=DOCS_RS");
    if env::var_os("DOCS_RS").is_some() {
        // docs.rs builds crates offline. The declarations in this crate do not
        // need a native library until a final binary is linked, so rustdoc must
        // not fetch and compile the reference implementation.
        return;
    }
    let manifest_dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let build_support = manifest_dir.join("build-support");
    println!("cargo:rerun-if-env-changed=FDK_AAC_SOURCE_DIR");
    println!("cargo:rerun-if-env-changed=FDK_AAC_REVISION");
    println!(
        "cargo:rerun-if-changed={}",
        build_support.join("test-bridge.patch").display()
    );
    let root = upstream_source();
    let cmake_lists = root.join("CMakeLists.txt");

    println!("cargo:rerun-if-changed={}", cmake_lists.display());
    println!(
        "cargo:rerun-if-changed={}",
        build_support.join("qmf-test-wrapper.bridge").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("libAACenc/include/aacenc_lib.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("libAACdec/include/aacdecoder_lib.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("libSYS/include/FDK_audio.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("libSYS/include/machine_type.h").display()
    );

    let cmake = fs::read_to_string(&cmake_lists).unwrap();
    let sources = extract_library_sources(&cmake);
    if sources.is_empty() {
        panic!("failed to extract fdk-aac source list from CMakeLists.txt");
    }

    let mut build = cc::Build::new();
    build.cpp(true);
    build.flag_if_supported("-std=c++11");
    build.flag_if_supported("-fno-exceptions");
    build.flag_if_supported("-fno-rtti");
    build.define("FDK_RUST_TEST_BRIDGE", None);

    for include in [
        "libAACdec/include",
        "libAACenc/include",
        "libSYS/include",
        "libArithCoding/include",
        "libDRCdec/include",
        "libSACdec/include",
        "libSACenc/include",
        "libSBRdec/include",
        "libSBRenc/include",
        "libMpegTPDec/include",
        "libMpegTPEnc/include",
        "libFDK/include",
        "libPCMutils/include",
    ] {
        build.include(root.join(include));
    }
    build.include(&root);

    for source in sources {
        let path = root.join(&source);
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(path);
    }
    let wrapper =
        PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("qmf_test_wrapper.cpp");
    fs::write(
        &wrapper,
        fs::read(build_support.join("qmf-test-wrapper.bridge")).expect("read QMF test bridge"),
    )
    .expect("write generated QMF test bridge");
    build.file(wrapper);

    build.compile("fdk-aac");
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
    let patch = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"))
        .join("build-support/test-bridge.patch");
    run(Command::new("git")
        .args(["-C"])
        .arg(&checkout)
        .args(["apply", "--whitespace=nowarn"])
        .arg(&patch));
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

fn extract_library_sources(cmake: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut in_sources = false;

    for line in cmake.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("set(") {
            in_sources = !trimmed.starts_with("set(fdk_aacinclude_HEADERS")
                && !trimmed.starts_with("set(libfdk_aac_SOURCES");
            continue;
        }

        if in_sources && trimmed == ")" {
            in_sources = false;
            continue;
        }

        let source = trimmed.trim_end_matches(')');
        if in_sources && source.starts_with("lib") && source.ends_with(".cpp") {
            out.push(source.to_owned());
        }
    }

    out
}
