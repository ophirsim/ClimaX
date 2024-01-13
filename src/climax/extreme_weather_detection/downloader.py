import subprocess

def download(url, destination_path=None, accept_list=None, reject_list=None):
    """
    Downloads all of the files from a given URL, at the specified path, while ignoring the specified files

    Arguments:
        - url: string -- the webpage url where the files are located online
        - destination_path: string -- the path where you would like the files to be downloaded into. By default this is your current working directory
        - accept_list: string -- string of comma separated regex snippets that must be in file for it to downloaded
        - reject_list: string -- string of comma separated regex snippets that cannot be in file for it to downloaded
    """
    command = ['wget', '-m', '-p', '-E', '-k', '-K', '-np', '-nd']

    if destination_path is not None:
        command.append('-P')
        command.append(destination_path)

    if accept_list is not None:
        command.append('-A')
        command.append(accept_list)

    if reject_list is not None:
        command.append('-R')
        command.append(reject_list)

    command.append(url)
    subprocess.run(command)