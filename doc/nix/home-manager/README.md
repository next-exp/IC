# Bootstrap home manager

## TLDR

If

+ nix is enabled for your user account
+ you have a network connection which is not absurdly slow

then you could have a personal home-manager installation and configuration up
and running in under two minutes, simply by pasting the following into a
bash-like shell:

```shell
# Decide where you want your home-manager config to live
HM_DIR=$HOME/my-home-manager

# Download and unpack the home-manager config template
cd /tmp
curl -L https://github.com/jacg/IC/tarball/manage-with-nix > IC.tgz
nix-shell -p pkgs.gnutar --run "tar xvf IC.tgz --wildcards '*/nix/home-manager' --strip-components=3"
mv home-manager $HM_DIR

# Tell home-manager where its config lives
mkdir -p $HOME/.config/nixpkgs
ln -s $HM_DIR/nixpkgs/home.nix $HOME/.config/nixpkgs/home.nix

# Bootstrap your personal home-manager installation and configuration
cd $HM_DIR
nix-shell bootstrap-home-manager
# You should also run `nix-shell bootstrap-home-manager` (from $HM_DIR) whenever
# you change the home-manager version in ./sources.nix
```

If this worked, then `home-manager` is ready: after changing your `home.nix` you
should be able to switch to the newly-specified configuration by running
`home-manager switch` (from *any* directory on your system):

```shell
# First edit $HM_DIR/nixpkgs/home.nix, then
home-manager switch
```

## I'd rather not copy 'n' paste blindly: Talk me through it!

+ Make sure that Nix is [installed and configured](../install-nix/README.md):

  ```shell
  nix-shell cowsay -p --run 'cowsay Looks like Nix is working.'
  ```
  Expect some noisy Nix output.

  If everything is working correctly, you should, eventually, see an ASCII-art
  picture of a cow saying that Nix is working. This means that you are ready to
  install home-manager in your user account. (If not, have you done
  [this](../install-nix/README.md)?)

1. Decide where you want your home-manager configuration files to live.

   ```shell
   HM_DIR=$HOME/my-home-manager
   ```

   I recommend that you let home-manager manage the configurations of most
   programs that you use in your user account, and that you keep all your
   configuration file sources version-controlled in this same directory. More on
   that, below.

2. Copy this directory (the one containing the README in which these words are
   written) into your personal space on your machine, at the location that you
   specified with the variable `HM_DIR` in step 1.

   ```shell
   cd /tmp
   curl -L https://github.com/jacg/IC/tarball/manage-with-nix > IC.tgz
   nix-shell -p pkgs.gnutar --run "tar xvf IC.tgz --wildcards '*/nix/home-manager' --strip-components=3"
   mv home-manager $HM_DIR
   ```

   [Aside: Note that we are using `nix-shell -p pkgs.gnutar` to ensure that
   (regardless of the default `tar` on your system) we will use GNU `tar` which
   provides options such as `--wildcards` and `--strip-components`.

   If your default `tar` provides these options, then you could run the `tar xvf
   IC.tgz ...` command directly, without wrapping it in `nix-shell -p
   pkgs.gnutar`. Using the `nix-shell` guarantees that this step will work
   correctly, on *any* nix-enabled system.]

3. Install `home-manager` for your user.

   Inside *your copy* of this directory, execute `bootstrap-home-manager` with
   `nix-shell`

   ```shell
   cd $HM_DIR
   nix-shell bootstrap-home-manager
   ```

   This step ephemerally installs `home-manager` into a `nix-shell` in which it
   instructs `home-manager` to install a persistent environment for your user:
   that environment contains `home-manager` itself, along with some examples of
   how to install and configure other packages.

   This step should be performed any time you change the version of
   `home-manager` you want to use. The `home-manager` version is specified in
   `sources.nix`.

4. Make sure that `home-manager` can find its configuration file in its default location.

   Create a symlink which makes your `home.nix` file appear at `$HOME/.config/nixpkgs/home.nix`.

   ```shell
   mkdir -p $HOME/.config/nixpkgs
   ln -s $HM_DIR/nixpkgs/home.nix $HOME/.config/nixpkgs/home.nix
   ```

Home-manager should now be ready to use by this user.

To check the installation:

1. Edit `$HM_DIR/nixpkgs/home.nix`, adding `figlet` to the
   packages listed in `home.packages`. (That is, just below `bat`.)

2. Update your environment with `home-manager switch`.

3. Verify that `figlet` has been installed with `figlet home-manager works!`

If you see "home-manager works!" written out in big letters, then home-manager
has been installed correctly, and you have just used it to install a package
(`figlet`) into your user's environment.

## Automatic environment switching with `direnv`

1. Hook `direnv` into your shell. Exactly how you do this depends on which shell
   you use. The details are described [here](https://direnv.net/docs/hook.html).

   In the case of `bash` it amounts to placing `eval "$(direnv hook bash)"` at
   the end of your `.bashrc`.

   Beware, Macs now use `zsh` by default: in the case of `zsh` you should add
   `eval "$(direnv hook zsh)"` to the end of your `.zshrc`.

2. Instruct `direnv` to use the Nix specification of the environment when
   entering `$ICTDIR`:

   ```shell
   echo use nix > $ICTDIR/.envrc
   ```

3. Give `direnv` permission to switch environment automatically when entering
   `$ICTDIR`:

   ```shell
   cd $ICTDIR
   direnv allow
   ```

   Two points to note:

   1. You can revoke this permission with `direnv deny`

   2. By default `direnv allow/deny` act on the current directory, but they both
      accept an optional path.

## This page is about `home-manager` so why are we talking about `direnv`?

1. `direnv` is useful, but it needs to be installed.

2. By default `nix-shell` takes somewhere between 3 and 7 seconds to enable the
   IC environment. This is very annoying, especially if you integrate direnv
   with your editor: switching buffer in your editor can be stalled for multiple
   seconds.

3. `nix-direnv` is a tool which makes this switching instantaneous, by caching
   previously seen environments. This also needs installation and configuration.

By using home-manager with the `home.nix` you were given here, installation and
integration of `direnv` and `nix-direnv` has been taken care of and tested for
you. Without `home-manager` you would have to carry out these steps by hand. You
might make a mistake, and you might require someone's help to sort it out. With
home-manager, the scope for errors and the concomitant waste of time has been
minimised.

## Why should I do this?

Home manager is NOT necessary for working with IC, but it can improve the
quality of your life. Among many other benefits, it will make it easier for
others to help you to ensure you have a properly working environment.

Now that the Nix package manager is a prerequisite for working on IC, you might
as well take advantage of the fact that Nix is available to you, and reap the
benefits that home-manager offers.

### What is home-manager?

Home manager uses Nix to manage the installation and configuration of software
that you use in your per-user working environment. Benefits include:

+ Install and uninstall software without admin privileges.

+ Try new packages with guarantees that you can revert to the previous state
  without anything breaking.

+ Configurations are reliable and reproducible.

+ Transferring your personal environment to a different machine is trivially
  easy (as long as Nix is available on that machine).

  The script at the top of this page can be adapted to use your personal
  home-manager configuration instead of the template.

  This means that **installing your environment on *ANY* nix-enabled machine is as
  simple as launching your script, and letting home-manager download and
  configure all the packages you are used to.**

### Ephemeral package installation with `nix-shell`

Nix makes it very easy to try out software packages without fear of breaking
anything in your existing environment. (With other package managers, installing
and uninstalling some package often leaves the system in a modified state.)

One particularly convenient way of trying a package in Nix is with ephemeral
installations using `nix-shell`.

As an example, let's take `lolcat` for a spin. Just like `cowsay` and `figlet`,
`lolcat` is a package which isn't terribly useful, but makes a good guinea-pig
for installation/uninstallation experiments. Install `lolcat` ephemerally and
try it out, with:

```shell
nix-shell -p lolcat # This will drop you in a shell where lolcat is available
ls -l / | lolcat    # A pretty, colourful listing should appear
```

Within this shell (and *only* this shell), you can play around with `lolcat` to
your heart's content: it adds pretty colours to whatever you pipe into it.

When you have seen enough, simply exit this shell, and `lolcat` disappears.
While you were in the shell, the rest of your system was unaware of the
existence of `lolcat`: it could not have interfered with anything else.

The first time you run `nix-shell -p lolcat` it is likely to take some time as
it might need to download `lolcat` and its dependencies. On subsequent
invocations, it is likely to run much faster, as everything that is needed has
been kept in the nix store on your machine ... at least until you instruct Nix
to collect garbage. You may want to run `nix-collect-garbage` if the Nix store
grows too large and you want to free up disk space wasted on packages (or older
versions) you are not using.

### Persistent package installation with `home-manager`

In the previous section we installed `lolcat` *ephemerally* with `nix-shell`:
the installed package was only available in a single shell, and disappeared as
soon as the shell was exited.

It is possible to install and remove packages *persistently* in your personal
environment by using `nix-env` to *mutate* your personal profile (you can think
of this as a personal `brew`, `apt-get`, `pacman`, `yum`, etc.). I do NOT
recommend this.

I suggest you use `home-manager` to manage your personal configuration
*declaratively*. The declarative approach makes it *much* easier to transfer
your environment to different machines, to understand what is present in your
environment, to share package configuration wisdom with your colleagues, to get
help from others, and much more.

If you have tried a package by installing it ephemerally with `nix-shell` and
found that it is useful to you, then you might want to make it persistently
available. With `home-manager` this is a two-step process:

+ Add the package to the `home.packages` list in your `home.nix`

+ Instruct `home-manager` to create and switch to an environment matching your
  new specification:

  ```shell
  home-manager switch
  ```

### Version control your environments

I recommend that you place your environment specification in version control:
```shell
git init $HM_DIR  # Assuming you set HM_DIR according to instructions above
```

You were instructed, above, to link your `home.nix` to
`$HOME/.config/nixpkgs/home.nix`. This is the simplest short-term way of getting
home-manager to work for you. However, I would recommend a more powerful and
all-encompassing approach for the long-term:

1. Place *all your personal configuration files/directories*

   - `$HOME/.bash{rc,_profile}`
   - `$HOME/.emacs.d`
   - `$HOME/.config/htop`
   - etc.

   into `$HM_DIR`

2. Instruct home-manager to make these files appear in their standard locations.
   See `$HM_DIR/{eg-ro,eg-rw,nixpkgs/home.nix}` for examples of this.

3. Keep track of changes with Git (or any VCS of your choice).

In this scheme, installing your personal environment on *any* Nix-enabled
machine amounts to no more than

1. `git clone <your-home-manager-repo> $HM_DIR`
2. `cd $HM_DIR`
3. `nix-shell bootstrap-home-manager`
4. `mkdir -p $HOME/.config/nixpkgs`
5. `ln -s $HM_DIR/nixpkgs/home.nix $HOME/.config/nixpkgs/home.nix`
6. DONE!

Note that all these steps can be wrapped into a single script, so **you could
install your whole environment with a single command!**

Caveat: when you take this approach, you must be careful not to commit any
*secrets* (e.g. private ssh keys) into a repository which you will ever place in
some publicly accessible locations (such as GitHub).
