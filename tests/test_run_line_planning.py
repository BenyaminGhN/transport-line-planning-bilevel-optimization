from scripts.run_line_planning import example_instance, main


def test_example_builds_complete_instance():
    instance = example_instance()
    assert instance.n_lines == 2
    assert instance.n_routes == 2
    assert instance.A_cap.shape == (4, 2)


def test_main_reports_matching_exact_and_ga_results(capsys):
    main()
    output = capsys.readouterr().out
    assert "Candidate lines:" in output
    assert "Exact enumeration:" in output
    assert "Genetic algorithm:" in output
    assert output.count("best x: 11") == 2
    assert "plans evaluated: 4" in output
    assert "gap to exact: 0.000000" in output
