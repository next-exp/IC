{ config, pkgs, ... }:
let
  link = config.lib.file.mkOutOfStoreSymlink;
in
with pkgs;
{
  home.packages = [
    (pkgs.callPackage ../sources.nix {}).home-manager
    # Add and remove packages in this list to suit your needs.
    bat # git-aware, intelligent, helpful, colourful version of `cat`
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

  # We've only scratched the surface of possibilities in home-manager ...

}
