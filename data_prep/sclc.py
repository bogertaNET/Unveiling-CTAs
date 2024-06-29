import concurrent.futures
import copy

import inflect
import regex as re


class SCLConverter:
    def __init__(self, data):
        self.data = data

        self.deepcopied = copy.deepcopy(data)
        self.eng = inflect.engine()
        self.c = 0
        self.pattern_file = re.compile(
            r"([' =]{0,1}(sftp:){0,1}(HKEY_[a-zA-Z_]+){0,1}(%APPDATA%){0,1}([a-z]:)?(\/|\\|\\\\)(((\\|\/|\\\\)?[^\x00-\x7F]?[a-z0-9^&'@{}\[\] ,$=!\-#\(\)%\.\+~_]{0,55}){1,3}(\\|\/|\\\\)){0,13}([^\\\/:\*\"<>\|\s]{0,55}(\.)?[a-z0-9]?))",
            re.I | re.U,
        )
        self.pattern_ip = re.compile(r"(((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.?\b){4})")
        self.pattern_domain = re.compile(
            r"((?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(com|net|co[.]uk|edu[.]cn|info|io|ca|org|local|biz|au|cn|uk|nl|de|ly))",
            re.I | re.U,
        )
        self.pattern_url = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            re.I,
        )
        self.pattern_longfiles = re.compile(
            r"( .*[\/\\])*[^\/\\]*\.((jpg)|(png)|(gif)|(pdf)|(doc)|(docx)|(xls)|(ps1)|(zip)|(rar)|(jpeg)|(xlsx)|(ppt)|(pptx))$",
            re.I,
        )
        self.sclc_run().obfuscate()

    def sclc_run(self):
        for ta in self.data.keys():
            for vic in self.data[ta].keys():
                victim = self.data[ta][vic]
                for cmd in victim:
                    ind = self.deepcopied[ta][vic].index(cmd)
                    if (
                        "initial beacon" in cmd["data"]
                        or "new_bot_instance" in cmd["data"]
                    ):
                        self.deepcopied[ta][vic].remove(cmd)
                        continue
                    if "download of" in cmd["data"]:
                        self.deepcopied[ta][vic].remove(cmd)
                        continue
                    if cmd["data"][:5] == "run: ":
                        self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                            vic
                        ][ind]["data"].replace("run: ", "execute_cmd: ")
                    if cmd["data"][:4] == "run ":
                        self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                            vic
                        ][ind]["data"].replace("run ", "execute_cmd: ")

                    if cmd["data"][:6] == "shell ":
                        self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                            vic
                        ][ind]["data"].replace("shell ", "execute_cmd: ")
                    if cmd["data"][:20] == "started download of ":
                        self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                            vic
                        ][ind]["data"].replace(
                            "started download of ", "download_file: "
                        )

                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace("execute_cmd: del ", "remove ")
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace("execute_cmd: DEL ", "remove ")
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace("rm ", "remove ")
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace("ls ", "list files in ")
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace("ls ", "list files in ")
                    self.deepcopied[ta][vic][ind]["data"] = (
                        self.deepcopied[ta][vic][ind]["data"]
                        .replace('"', "")
                        .replace("(", "")
                        .replace(")", "")
                    )

                    if "ls" == cmd["data"]:
                        self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                            vic
                        ][ind]["data"].replace("ls", "list files in .")

                    if "sleep for " == cmd["data"][:11]:
                        pass

                    elif "sleep " == cmd["data"][:6]:
                        arr = cmd["data"].split(" ")
                        if len(arr) == 2:
                            self.deepcopied[ta][vic][ind][
                                "data"
                            ] = f"sleep for {arr[1]}s"
                        elif len(arr) == 3 and arr[1] != "for":
                            self.deepcopied[ta][vic][ind][
                                "data"
                            ] = f"sleep for {arr[1]}s ({arr[2]}% jitter)"

                    if "steal_token" == cmd["data"][:11]:
                        arr = cmd["data"].split(" ")
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = f"steal token from PID {arr[1]}"

                    if "ps" == cmd["data"]:
                        self.deepcopied[ta][vic][ind]["data"] = f"list processes"

                    if "net domain" == cmd["data"]:
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = f"execute_cmd: net domain"

                    if "hashdump" == cmd["data"]:
                        self.deepcopied[ta][vic][ind]["data"] = "dump hashes"

                    if "make_token" == cmd["data"][:10]:
                        arr = cmd["data"].split(" ")
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = f"create a token for {arr[1]}"

                    if "rev2self" == cmd["data"]:
                        self.deepcopied[ta][vic][ind]["data"] = "revert token"

                    if "jobs" == cmd["data"]:
                        self.deepcopied[ta][vic][ind]["data"] = "list jobs"

                    if "net share" == cmd["data"][:9]:
                        arr = cmd["data"].split(" ")
                        arr[2] = arr[2].replace("\\\\", "")
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = f"execute_cmd: net share on {arr[2]}"

                    if "mkdir" == cmd["data"][:5]:
                        arr = cmd["data"].split(" ")
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = f"make directory {arr[1]}"

                    if "pwd" == cmd["data"]:
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = "print working directory"

                    if "ppid" == cmd["data"][:4]:
                        arr = cmd["data"].split(" ")
                        self.deepcopied[ta][vic][ind][
                            "data"
                        ] = f"spoof {arr[1]} as parent process"

                    if "upload" == cmd["data"][:6]:
                        self.deepcopied[ta][vic][ind]["data"] = (
                            self.deepcopied[ta][vic][ind]["data"]
                            .replace("(", "")
                            .replace(")", "")
                        )
                        if " as " in self.deepcopied[ta][vic][ind]["data"]:
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(" as ", " ")
                    if "/outfile:" in cmd["data"]:
                        self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                            vic
                        ][ind]["data"].replace("/outfile:", "/outfile: ")

                    if "mimikatz" in cmd["data"]:
                        if "/domain" in cmd["data"]:
                            domain = cmd["data"].split("/domain:")[1].split(" ")[0]
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(domain, "DOMAIN")
                        if "/dc:" in cmd["data"]:
                            dc = cmd["data"].split("/dc:")[1].split(" ")[0]
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(dc, "DC_DOMAIN")
                        if "/user:" in cmd["data"]:
                            user = cmd["data"].split("/user:")[1].split(" ")[0]
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(user, "USER")
                        if "/authuser:" in cmd["data"]:
                            authuser = cmd["data"].split("/authuser:")[1].split(" ")[0]
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(authuser, "AUTHUSER")
                        if "/authdomain:" in cmd["data"]:
                            authdomain = (
                                cmd["data"].split("/authdomain:")[1].split(" ")[0]
                            )
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(authdomain, "AUTHDOMAIN")
                        if "/authpassword:" in cmd["data"]:
                            authpassword = (
                                cmd["data"].split("/authpassword:")[1].split(" ")[0]
                            )
                            self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][
                                vic
                            ][ind]["data"].replace(authpassword, "AUTHPASSWORD")

        for ta in self.deepcopied.keys():
            for vic in self.deepcopied[ta].keys():
                new_vic_list = []

                for cmd in self.deepcopied[ta][vic]:
                    if cmd not in new_vic_list:
                        new_vic_list.append(cmd)
                self.deepcopied[ta][vic] = new_vic_list

        for ta in self.deepcopied.keys():
            for vic in self.deepcopied[ta].keys():
                for cmd in self.deepcopied[ta][vic]:
                    ind = self.deepcopied[ta][vic].index(cmd)
                    if "sleep for" in cmd["data"]:
                        arr = cmd["data"].split(" ")
                        newstr = "sleep for "
                        sec = self.eng.number_to_words(arr[2].replace("s", ""))
                        newstr += sec + " seconds"
                        if len(arr) > 3:
                            jitter = self.eng.number_to_words(arr[3].replace("%", ""))
                            newstr += f" ({jitter}% jitter)"
                        self.deepcopied[ta][vic][ind]["data"] = newstr
        to_be_dropped = []
        for ta in self.deepcopied.keys():
            for vic in self.deepcopied[ta].keys():
                if len(self.deepcopied[ta][vic]) == 0:
                    to_be_dropped.append((ta, vic))
        for each in to_be_dropped:
            del self.deepcopied[each[0]][each[1]]

        dicto = {}
        for ta in self.deepcopied.keys():
            for victim in self.deepcopied[ta].keys():
                if victim not in dicto:
                    dicto[victim] = []
                dicto[victim].append((ta, self.deepcopied[ta][victim]))
        for victim in dicto.keys():
            uniques = []
            if len(dicto[victim]) > 1:
                for each in dicto[victim]:
                    if each[1] not in uniques:
                        uniques.append(each[1])
                    else:
                        del self.deepcopied[each[0]][victim]

        return self

    def replace_ips(self, ta, vic, cmd, ind):
        ips = self.pattern_ip.findall(cmd["data"])
        if len(ips) > 0:
            for ip in ips:
                try:
                    original = ip[0]
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace(original, f"IPADDRESS")
                    urls = self.pattern_url.findall(
                        self.deepcopied[ta][vic][ind]["data"]
                    )
                    if len(urls) > 0:
                        for url in urls:
                            self.deepcopied[ta][vic][ind]["data"] = (
                                self.pattern_url.sub(
                                    "IPADDR_URL", self.deepcopied[ta][vic][ind]["data"]
                                )
                            )
                except Exception as e:
                    print(e, ip)
        return self

    def replace_domains(self, ta, vic, cmd, ind):
        domains = self.pattern_domain.findall(cmd["data"])
        if len(domains) > 0:
            for domain in domains:
                try:
                    original = domain[0]
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace(original, f"DOMAIN")
                    urls = self.pattern_url.findall(
                        self.deepcopied[ta][vic][ind]["data"]
                    )
                    if len(urls) > 0:
                        for _ in urls:
                            self.deepcopied[ta][vic][ind]["data"] = (
                                self.pattern_url.sub(
                                    "DOMAIN_URL", self.deepcopied[ta][vic][ind]["data"]
                                )
                            )
                except Exception as e:
                    print(e, domain)
        return self

    def replace_paths(self, ta, vic, cmd, ind):
        paths = self.pattern_file.findall(cmd["data"])
        if len(paths) > 0:
            for path in paths:
                if "http://" in path[0] or "https://" in path[0]:
                    continue
                try:
                    original = path[0]
                    if "p://" in original and (
                        "http://" in cmd["data"] or "https://" in cmd["data"]
                    ):
                        continue
                    self.deepcopied[ta][vic][ind]["data"] = self.deepcopied[ta][vic][
                        ind
                    ]["data"].replace(original, f" FPATH")
                except Exception as e:
                    print(e, path)
        return self

    def replace_long_paths(self, ta, vic, cmd, ind):
        long_paths = self.pattern_longfiles.findall(cmd["data"], timeout=120)
        if len(long_paths) > 0:
            for path in long_paths:
                try:
                    original = path[0]
                    self.deepcopied[ta][vic][ind]["data"] = self.pattern_longfiles.sub(
                        " FPATH", self.deepcopied[ta][vic][ind]["data"]
                    )
                except Exception as e:
                    print(e, path)
        return self

    def replace(self, ta, vic, cmd):
        ind = self.deepcopied[ta][vic].index(cmd)
        self.replace_ips(ta, vic, cmd, ind)
        self.replace_domains(ta, vic, cmd, ind)
        self.replace_long_paths(ta, vic, cmd, ind)
        self.replace_paths(ta, vic, cmd, ind)

        return self

    def obfuscate(self):
        total_len = 0
        for ta in self.deepcopied.keys():
            for vic in self.deepcopied[ta].keys():
                total_len += len(self.deepcopied[ta][vic])

        for ta in self.deepcopied.keys():
            for vic in self.deepcopied[ta].keys():
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as e:
                    fut = [
                        e.submit(self.replace, ta, vic, cmd)
                        for cmd in self.deepcopied[ta][vic]
                    ]
                    try:
                        for r in concurrent.futures.as_completed(fut, timeout=80):
                            r.result(timeout=20)
                            self.c += 1
                    except Exception as ew:
                        print(ew)
        return self

    def print_same(self):
        for ta in self.deepcopied.keys():
            for vic in self.deepcopied[ta].keys():
                prev = ""
                for cmd in self.deepcopied[ta][vic]:
                    if prev == "":
                        prev = cmd
                        continue
                    if cmd["when"] == prev["when"]:
                        print(vic, prev, cmd)
                        print()
                        print()
                    prev = cmd
        return self
