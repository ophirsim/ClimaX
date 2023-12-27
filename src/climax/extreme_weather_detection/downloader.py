import subprocess

def download(url, destination_path=None, ext=None):
    """
    Downloads all of the files from a given URL, at the specified path, while ignoring the specified files

    Arguments:
        - url: string -- the webpage url where the files are located online
        - destination_path: string -- the path where you would like the files to be downloaded into. By default this is your current working directory
        - ext: string -- the extension type for the files you would like to download, all other extensions are ingored. By default, all extension types are downloaded
    """
    command = ['wget', '-r', '--no-parent']
    command = ['wget', '-m', '-p', '-E', '-k', '-K', '-np', '-nd']

    if destination_path is not None:
        command.append('-P')
        command.append(destination_path)

    if ext is not None:
        command.append('-A')
        command.append(ext)

    command.append(url)

    # print(command)
    subprocess.run(command)