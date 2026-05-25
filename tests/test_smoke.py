from thesis_project.main import main


def test_main_runs(capsys):
    main()
    captured = capsys.readouterr()
    assert "set up" in captured.out
