manage.sh()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    prev2="${COMP_WORDS[COMP_CWORD-2]}"


    case "${prev}" in
        manage.sh)
                        local opts="make_environment\
                                    run_tests\
                                    run_tests_par\
                                    compile_and_test\
                                    compile_and_test_par\
                                    download_test_db\
                                    clean"
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        *)
        ;;
    esac

    if [[ ${prev2} == "manage.sh" ]] ; then
            case "${prev}" in
                        install_and_check|install|work_in_python_version|make_environment)
                        local versions="2.7 3.5 3.6"
                    COMPREPLY=( $(compgen -W "${versions}" -- ${cur}) )
                    return 0
                    ;;
                *)
                ;;
            esac
    fi

        return 0
}


source_manage.sh()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    prev2="${COMP_WORDS[COMP_CWORD-2]}"


    case "${prev}" in
        manage.sh)
                        local opts="install_and_check\
                                    install\
                                    work_in_python_version\
                                    make_environment\
                                    run_tests\
                                    run_tests_par\
                                    compile_and_test\
                                    compile_and_test_par\
                                    download_test_db\
                                    clean"
            COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
            return 0
            ;;
        *)
        ;;
    esac

    if [[ ${prev2} == "manage.sh" ]] ; then
            case "${prev}" in
                        install_and_check|install|work_in_python_version|make_environment)
                        local versions="2.7 3.5 3.6"
                    COMPREPLY=( $(compgen -W "${versions}" -- ${cur}) )
                    return 0
                    ;;
                *)
                ;;
            esac
    fi

        return 0
}

complete -F manage.sh -o bashdefault -o default bash
complete -F source_manage.sh -o bashdefault -o default source
complete -F source_manage.sh -o bashdefault -o default .
