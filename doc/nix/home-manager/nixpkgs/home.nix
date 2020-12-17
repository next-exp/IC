{ config
, pkgs ? import (fetchTarball "nixpkgs=https://github.com/NixOS/nixpkgs-channels/archive/2f6440eb09b7e6e3322720ac91ce7e2cdeb413f9.tar.gz") {}
, ... }:
let
  sources = (pkgs.callPackage ../sources.nix { pkgs = pkgs; });
  link = config.lib.file.mkOutOfStoreSymlink;
in
{
  home.packages = [
    sources.home-manager
    # Add and remove packages in this list to suit your needs.
    pkgs.bat # git-aware, intelligent, helpful, colourful version of `cat`
    pkgs.fd  # better find
    pkgs.ripgrep # better grep / grep-find
    pkgs.ripgrep-all # Enable ripgrep in PDFs, ebooks, doc, zip, tar.gz, etc.
    pkgs.du-dust # better du
    pkgs.tldr # simplified man-pages
  ];

  # Some programs require/permit more intricate configuration
  programs.direnv.enable = true;
  programs.direnv.enableNixDirenvIntegration = true;

  # Sick and tired of making git commits with author email
  # 'Pepito@Pepitos-Macbook-Pro'? Then why not let home-manager take care of
  # configuring Git for you? Adapt the following to your needs:
  programs.git = {
    enable = true;
  # Local .gitconfig files may override these settings:
  #   userName = "Fulana Menganez-Perenganez";
  #   userEmail = "fulana@ific.es";
  #   aliases = {
  #     lg  = "log --graph --decorate --oneline";
  #     lga = "log --graph --decorate --oneline --all";
  #   };
  #   lfs.enable = true;
  };

  # This is how to get home-manager to place your version-controlled
  # configuration files in your home directory. There are two approaches,
  # read-only and read-write. Read the text in `eg-ro` and `eg-rw` for more
  # details. Replace these with your own real-world cases.
  home.file.".example-read-only" .source =      ../eg-ro;
  home.file.".example-read-write".source = link ../eg-rw;
  # The previous two lines are just examples, which you can adapt to your own
  # needs. The next line is vital: it ensures that home-manager will be able to
  # find its configuration file (that's the file in which these words are
  # written) in the standard location.
  home.file.".config/nixpkgs".source = link ../nixpkgs;

  # We've only scratched the surface of possibilities in home-manager ...
}
