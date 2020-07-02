{ pkgs ? import <nixpkgs> {} }:
with pkgs;
{
  home-manager = let
    src = builtins.fetchGit {
      name = "home-manager-2020-06-29";
      url = https://github.com/rycee/home-manager;
      rev = "7f7348b47049e8d25fb5b98db1d6215f8f643f0d";
    };
  # `path` is required for `home-manager` to find its own sources
  in callPackage "${src}/home-manager" { path = "${src}"; };
}
