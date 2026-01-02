use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

type DynError = Box<dyn std::error::Error>;

fn main() {
    if let Err(e) = try_main() {
        eprintln!("{}", e);
        std::process::exit(-1);
    }
}

fn try_main() -> Result<(), DynError> {
    let task = env::args().nth(1);
    match task.as_deref() {
        Some("dist") => dist()?,
        _ => print_help(),
    }
    Ok(())
}

fn print_help() {
    eprintln!(
        "Tasks:

dist            builds application and copy dll as matlab mex files
"
    )
}

fn dist() -> Result<(), DynError> {
    let _ = fs::remove_dir_all(dist_dir());
    fs::create_dir_all(dist_dir())?;

    dist_binary()?;

    Ok(())
}

fn dist_binary() -> Result<(), DynError> {
    let extension = match std::env::consts::OS {
        "windows" => ".mexw64",
        "linux" => ".mexa64",
        "macos" => panic!("Target macos are currently unsupported."),
        unsupported_target => panic!("Target {unsupported_target} are currently unsupported."),
    };

    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let status = Command::new(cargo)
        .current_dir(project_root())
        .args(["build", "--release"])
        .status()?;

    if !status.success() {
        Err("cargo build failed")?;
    }

    let dst = project_root().join("target/release");

    if dst.is_dir() {
        for entry in (fs::read_dir(dst)?).flatten() {
            let path = entry.path();
            if !path.is_file() || path.extension().unwrap_or(OsStr::new("")) != "dll" {
                continue;
            }
            let filename = path
                .file_name()
                .unwrap_or(OsStr::new(""))
                .to_str()
                .unwrap_or("")
                .rsplit_once('.')
                .unwrap_or(("", ""))
                .0;
            fs::copy(&path, dist_dir().join(format!("{}{}", filename, extension)))?;
        }
    } else {
        panic!("Target path is not a directory")
    }

    Ok(())
}

fn project_root() -> PathBuf {
    Path::new(&env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(1)
        .unwrap()
        .to_path_buf()
}

fn dist_dir() -> PathBuf {
    project_root().join("dist")
}
