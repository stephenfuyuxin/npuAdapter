注，链接已经失效，对应的镜像仓/镜像已下架，仅作为记录，

# hiascend 昇腾镜像仓
```shell
https://www.hiascend.com/developer/ascendhub/detail/2761cb22f4da41e8be16891916f810a9
```

# 镜像
镜像名称格式：`<Image_name>:<Image_tag>`
```shell
swr.cn-south-1.myhuaweicloud.com/ascendhub/rerank-nim-800i:v4-arm64
```

# docker run
## 参考启动命令
```shell
docker run -e ASCEND_VISIBLE_DEVICES=0 -itd -u root --name=nim-rerank --net=host \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
-v /home/model:/model \
-e http_proxy=<ip:port> \
-e https_proxy=<ip:port> \
rerank_nim_800i:v2-arrch64  BAAI/bge-reranker-large 127.0.0.1 8080
```
## 实际启动命令
```shell
docker run -itd -u root --name=nim-rerank-npuid5 --net=host \
--device=/dev/davinci5 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
-v /etc/ascend_install.info:/etc/ascend_install.info \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
-v /the/path/of/model:/model \
swr.cn-south-1.myhuaweicloud.com/ascendhub/rerank-nim-800i:v4-arm64 BAAI/bge-reranker-large 127.0.0.1 8080
```
- 当镜像的 /model 目录下不存在指定的模型目录时（以 BAAI/bge-reranker-large 为例，目录名称为 bge-reranker-large），会从 hugging face 网站自动拉取指定模型，请保持网络通畅。这里提前下载好开源权重，并放置好路径，通过 -v 本地持久化到容器；
- 如果使用默认方式启动，image id、listen ip、listen port 都需要使用默认方式；
- ASCEND_VISIBLE_DEVICES 环境变量表示将宿主机上的 npu 卡挂载到容器，如果挂载多张卡使用逗号分隔，如：ASCEND_VISIBLE_DEVICES=0,1,2,3。这里未安装 ascend-docker-runtime ，通过 --device 方式挂载卡到容器使用；
- 启动容器会默认自动选择 npu 卡运行，如果用户想指定卡运行，则在启动命令中加入 -e TEI_NPU_DEVICE=指定的卡号（在ASCEND_VISIBLE_DEVICES范围内）。如上，不涉及；

# 开源权重
提前获取 BAAI/bge-reranker-large 开源权重，
```shell
# git lfs clone https://huggingface.co/BAAI/bge-reranker-large/ &
```
开源权重绝对路径，
```shell
/the/path/of/model/bge-reranker-large
```

# 运行并调测
启动 docker run 运行容器，

通过 docker logs 观察容器是否 ready ，看到 “Ready” 关键字说明容器已正常启动
```shell
# docker logs nim-rerank-npuid5
# docker exec -it nim-rerank-npuid5 /bin/bash

curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

# apt list --installed
```shell
# apt list --installed
Listing... Done
adduser/jammy,now 3.118ubuntu5 all [installed]
apt/jammy-updates,now 2.4.12 arm64 [installed]
base-files/jammy-updates,now 12ubuntu4.6 arm64 [installed]
base-passwd/jammy,now 3.5.52build1 arm64 [installed]
bash/jammy-updates,jammy-security,now 5.1-6ubuntu1.1 arm64 [installed]
binutils-aarch64-linux-gnu/jammy-updates,jammy-security,now 2.38-4ubuntu2.6 arm64 [installed,automatic]
binutils-common/jammy-updates,jammy-security,now 2.38-4ubuntu2.6 arm64 [installed,automatic]
binutils/jammy-updates,jammy-security,now 2.38-4ubuntu2.6 arm64 [installed,automatic]
bsdutils/jammy-updates,jammy-security,now 1:2.37.2-4ubuntu3.4 arm64 [installed]
bzip2/jammy,now 1.0.8-5build1 arm64 [installed,automatic]
ca-certificates/jammy-updates,jammy-security,now 20230311ubuntu0.22.04.1 all [installed,automatic]
coreutils/jammy-updates,now 8.32-4.1ubuntu1.2 arm64 [installed]
cpp-11/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
cpp/jammy,now 4:11.2.0-1ubuntu1 arm64 [installed,automatic]
curl/jammy-updates,jammy-security,now 7.81.0-1ubuntu1.16 arm64 [installed]
dash/jammy,now 0.5.11+git20210903+057cd650a4ed-3build1 arm64 [installed]
debconf/jammy,now 1.5.79ubuntu1 all [installed]
debianutils/jammy,now 5.5-1ubuntu2 arm64 [installed]
diffutils/jammy,now 1:3.8-0ubuntu2 arm64 [installed]
dos2unix/jammy,now 7.4.2-2 arm64 [installed]
dpkg/jammy-updates,now 1.21.1ubuntu2.3 arm64 [installed]
e2fsprogs/jammy-updates,jammy-security,now 1.46.5-2ubuntu1.1 arm64 [installed]
findutils/jammy,now 4.8.0-1ubuntu3 arm64 [installed]
fontconfig-config/jammy,now 2.13.1-4.2ubuntu5 all [installed,automatic]
fonts-dejavu-core/jammy,now 2.37-2build1 all [installed,automatic]
g++-11/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
g++/jammy,now 4:11.2.0-1ubuntu1 arm64 [installed]
gcc-11-base/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
gcc-11/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
gcc-12-base/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed]
gcc/jammy,now 4:11.2.0-1ubuntu1 arm64 [installed]
git-lfs/jammy-updates,jammy-security,now 3.0.2-1ubuntu0.2 arm64 [installed]
git-man/jammy-updates,jammy-security,now 1:2.34.1-1ubuntu1.11 all [installed,automatic]
git/jammy-updates,jammy-security,now 1:2.34.1-1ubuntu1.11 arm64 [installed]
gpgv/jammy-updates,jammy-security,now 2.2.27-3ubuntu2.1 arm64 [installed]
grep/jammy,now 3.7-1build1 arm64 [installed]
gzip/jammy-updates,now 1.10-4ubuntu4.1 arm64 [installed]
hostname/jammy,now 3.23ubuntu2 arm64 [installed]
init-system-helpers/jammy,now 1.62 all [installed]
less/jammy-updates,jammy-security,now 590-1ubuntu0.22.04.3 arm64 [installed,automatic]
libacl1/jammy,now 2.3.1-1 arm64 [installed]
libapt-pkg6.0/jammy-updates,now 2.4.12 arm64 [installed]
libasan6/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
libatomic1/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
libattr1/jammy,now 1:2.5.1-1build1 arm64 [installed]
libaudit-common/jammy,now 1:3.0.7-1build1 all [installed]
libaudit1/jammy,now 1:3.0.7-1build1 arm64 [installed]
libbinutils/jammy-updates,jammy-security,now 2.38-4ubuntu2.6 arm64 [installed,automatic]
libblkid1/jammy-updates,jammy-security,now 2.37.2-4ubuntu3.4 arm64 [installed]
libbrotli1/jammy,now 1.0.9-2build6 arm64 [installed,automatic]
libbsd0/jammy,now 0.11.5-1 arm64 [installed,automatic]
libbz2-1.0/jammy,now 1.0.8-5build1 arm64 [installed]
libc-bin/now 2.35-0ubuntu3.7 arm64 [installed,upgradable to: 2.35-0ubuntu3.8]
libc-dev-bin/jammy-updates,jammy-security,now 2.35-0ubuntu3.8 arm64 [installed,automatic]
libc-devtools/jammy-updates,jammy-security,now 2.35-0ubuntu3.8 arm64 [installed,automatic]
libc6-dev/jammy-updates,jammy-security,now 2.35-0ubuntu3.8 arm64 [installed,automatic]
libc6/jammy-updates,jammy-security,now 2.35-0ubuntu3.8 arm64 [installed]
libcap-ng0/jammy,now 0.7.9-2.2build3 arm64 [installed]
libcap2/jammy-updates,jammy-security,now 1:2.44-1ubuntu0.22.04.1 arm64 [installed]
libcbor0.8/jammy,now 0.8.0-2ubuntu1 arm64 [installed,automatic]
libcc1-0/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
libcom-err2/jammy-updates,jammy-security,now 1.46.5-2ubuntu1.1 arm64 [installed]
libcrypt-dev/jammy,now 1:4.4.27-1 arm64 [installed,automatic]
libcrypt1/jammy,now 1:4.4.27-1 arm64 [installed]
libctf-nobfd0/jammy-updates,jammy-security,now 2.38-4ubuntu2.6 arm64 [installed,automatic]
libctf0/jammy-updates,jammy-security,now 2.38-4ubuntu2.6 arm64 [installed,automatic]
libcurl3-gnutls/jammy-updates,jammy-security,now 7.81.0-1ubuntu1.16 arm64 [installed,automatic]
libcurl4/jammy-updates,jammy-security,now 7.81.0-1ubuntu1.16 arm64 [installed,automatic]
libdb5.3/jammy,now 5.3.28+dfsg1-0.8ubuntu3 arm64 [installed]
libdebconfclient0/jammy,now 0.261ubuntu1 arm64 [installed]
libdeflate0/jammy,now 1.10-2 arm64 [installed,automatic]
libdpkg-perl/jammy-updates,now 1.21.1ubuntu2.3 all [installed,automatic]
libedit2/jammy,now 3.1-20210910-1build1 arm64 [installed,automatic]
liberror-perl/jammy,now 0.17029-1 all [installed,automatic]
libexpat1-dev/jammy-updates,jammy-security,now 2.4.7-1ubuntu0.3 arm64 [installed,automatic]
libexpat1/jammy-updates,jammy-security,now 2.4.7-1ubuntu0.3 arm64 [installed,automatic]
libext2fs2/jammy-updates,jammy-security,now 1.46.5-2ubuntu1.1 arm64 [installed]
libffi8/jammy,now 3.4.2-4 arm64 [installed]
libfido2-1/jammy,now 1.10.0-1 arm64 [installed,automatic]
libfile-fcntllock-perl/jammy,now 0.22-3build7 arm64 [installed,automatic]
libfontconfig1/jammy,now 2.13.1-4.2ubuntu5 arm64 [installed,automatic]
libfreetype6/jammy-updates,jammy-security,now 2.11.1+dfsg-1ubuntu0.2 arm64 [installed,automatic]
libgcc-11-dev/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
libgcc-s1/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed]
libgcrypt20/jammy,now 1.9.4-3ubuntu3 arm64 [installed]
libgd3/jammy,now 2.3.0-2ubuntu2 arm64 [installed,automatic]
libgdbm-compat4/jammy,now 1.23-1 arm64 [installed,automatic]
libgdbm6/jammy,now 1.23-1 arm64 [installed,automatic]
libglib2.0-0/jammy-updates,jammy-security,now 2.72.4-0ubuntu2.3 arm64 [installed,automatic]
libglib2.0-data/jammy-updates,jammy-security,now 2.72.4-0ubuntu2.3 all [installed,automatic]
libgmp10/jammy,now 2:6.2.1+dfsg-3ubuntu1 arm64 [installed]
libgnutls30/jammy-updates,jammy-security,now 3.7.3-4ubuntu1.5 arm64 [installed]
libgomp1/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
libgpg-error0/jammy,now 1.43-3 arm64 [installed]
libgpm2/jammy,now 1.20.7-10build1 arm64 [installed,automatic]
libgssapi-krb5-2/jammy-updates,jammy-security,now 1.19.2-2ubuntu0.3 arm64 [installed]
libhogweed6/jammy,now 3.7.3-1build2 arm64 [installed]
libhwasan0/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
libicu70/jammy,now 70.1-2 arm64 [installed,automatic]
libidn2-0/jammy,now 2.3.2-2build1 arm64 [installed]
libisl23/jammy,now 0.24-2build1 arm64 [installed,automatic]
libitm1/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
libjbig0/jammy-updates,jammy-security,now 2.1-3.1ubuntu0.22.04.1 arm64 [installed,automatic]
libjpeg-turbo8/jammy,now 2.1.2-0ubuntu1 arm64 [installed,automatic]
libjpeg8/jammy,now 8c-2ubuntu10 arm64 [installed,automatic]
libk5crypto3/jammy-updates,jammy-security,now 1.19.2-2ubuntu0.3 arm64 [installed]
libkeyutils1/jammy,now 1.6.1-2ubuntu3 arm64 [installed]
libkrb5-3/jammy-updates,jammy-security,now 1.19.2-2ubuntu0.3 arm64 [installed]
libkrb5support0/jammy-updates,jammy-security,now 1.19.2-2ubuntu0.3 arm64 [installed]
libldap-2.5-0/jammy-updates,now 2.5.18+dfsg-0ubuntu0.22.04.2 arm64 [installed,automatic]
libldap-common/jammy-updates,now 2.5.18+dfsg-0ubuntu0.22.04.2 all [installed,automatic]
liblocale-gettext-perl/jammy,now 1.07-4build3 arm64 [installed,automatic]
liblsan0/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
liblz4-1/jammy,now 1.9.3-2build2 arm64 [installed]
liblzma5/jammy,now 5.2.5-2ubuntu1 arm64 [installed]
libmd0/jammy,now 1.0.4-1build1 arm64 [installed,automatic]
libmount1/jammy-updates,jammy-security,now 2.37.2-4ubuntu3.4 arm64 [installed]
libmpc3/jammy,now 1.2.1-2build1 arm64 [installed,automatic]
libmpdec3/jammy,now 2.5.1-2build2 arm64 [installed,automatic]
libmpfr6/jammy,now 4.1.0-3build3 arm64 [installed,automatic]
libncurses6/jammy-updates,jammy-security,now 6.3-2ubuntu0.1 arm64 [installed]
libncursesw6/jammy-updates,jammy-security,now 6.3-2ubuntu0.1 arm64 [installed]
libnettle8/jammy,now 3.7.3-1build2 arm64 [installed]
libnghttp2-14/jammy-updates,jammy-security,now 1.43.0-1ubuntu0.2 arm64 [installed,automatic]
libnsl-dev/jammy,now 1.3.0-2build2 arm64 [installed,automatic]
libnsl2/jammy,now 1.3.0-2build2 arm64 [installed]
libp11-kit0/jammy,now 0.24.0-6build1 arm64 [installed]
libpam-modules-bin/jammy-updates,jammy-security,now 1.4.0-11ubuntu2.4 arm64 [installed]
libpam-modules/jammy-updates,jammy-security,now 1.4.0-11ubuntu2.4 arm64 [installed]
libpam-runtime/jammy-updates,jammy-security,now 1.4.0-11ubuntu2.4 all [installed]
libpam0g/jammy-updates,jammy-security,now 1.4.0-11ubuntu2.4 arm64 [installed]
libpcre2-8-0/jammy-updates,jammy-security,now 10.39-3ubuntu0.1 arm64 [installed]
libpcre3/jammy-updates,jammy-security,now 2:8.39-13ubuntu0.22.04.1 arm64 [installed]
libperl5.34/jammy-updates,jammy-security,now 5.34.0-3ubuntu1.3 arm64 [installed,automatic]
libpng16-16/jammy,now 1.6.37-3build5 arm64 [installed,automatic]
libprocps8/jammy-updates,jammy-security,now 2:3.3.17-6ubuntu2.1 arm64 [installed]
libprotobuf-dev/jammy-updates,jammy-security,now 3.12.4-1ubuntu7.22.04.1 arm64 [installed]
libprotobuf-lite23/jammy-updates,jammy-security,now 3.12.4-1ubuntu7.22.04.1 arm64 [installed,automatic]
libprotobuf23/jammy-updates,jammy-security,now 3.12.4-1ubuntu7.22.04.1 arm64 [installed,automatic]
libprotoc23/jammy-updates,jammy-security,now 3.12.4-1ubuntu7.22.04.1 arm64 [installed,automatic]
libpsl5/jammy,now 0.21.0-1.2build2 arm64 [installed,automatic]
libpython3.10-dev/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed,automatic]
libpython3.10-minimal/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed,automatic]
libpython3.10-stdlib/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed,automatic]
libpython3.10/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed,automatic]
libreadline8/jammy,now 8.1.2-1 arm64 [installed,automatic]
librtmp1/jammy,now 2.4+20151223.gitfa8646d.1-2build4 arm64 [installed,automatic]
libsasl2-2/jammy-updates,now 2.1.27+dfsg2-3ubuntu1.2 arm64 [installed,automatic]
libsasl2-modules-db/jammy-updates,now 2.1.27+dfsg2-3ubuntu1.2 arm64 [installed,automatic]
libsasl2-modules/jammy-updates,now 2.1.27+dfsg2-3ubuntu1.2 arm64 [installed,automatic]
libseccomp2/jammy,now 2.5.3-2ubuntu2 arm64 [installed]
libselinux1/jammy,now 3.3-1build2 arm64 [installed]
libsemanage-common/jammy,now 3.3-1build2 all [installed]
libsemanage2/jammy,now 3.3-1build2 arm64 [installed]
libsepol2/jammy,now 3.3-1build1 arm64 [installed]
libsmartcols1/jammy-updates,jammy-security,now 2.37.2-4ubuntu3.4 arm64 [installed]
libsodium23/jammy,now 1.0.18-1build2 arm64 [installed,automatic]
libsqlite3-0/jammy-updates,jammy-security,now 3.37.2-2ubuntu0.3 arm64 [installed,automatic]
libss2/jammy-updates,jammy-security,now 1.46.5-2ubuntu1.1 arm64 [installed]
libssh-4/jammy-updates,jammy-security,now 0.9.6-2ubuntu0.22.04.3 arm64 [installed,automatic]
libssl-dev/jammy-updates,jammy-security,now 3.0.2-0ubuntu1.17 arm64 [installed]
libssl3/jammy-updates,jammy-security,now 3.0.2-0ubuntu1.17 arm64 [installed]
libstdc++-11-dev/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
libstdc++6/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed]
libsystemd0/jammy-updates,now 249.11-0ubuntu3.12 arm64 [installed]
libtasn1-6/jammy,now 4.18.0-4build1 arm64 [installed]
libtiff5/jammy-updates,jammy-security,now 4.3.0-6ubuntu0.9 arm64 [installed,automatic]
libtinfo6/jammy-updates,jammy-security,now 6.3-2ubuntu0.1 arm64 [installed]
libtirpc-common/jammy-updates,jammy-security,now 1.3.2-2ubuntu0.1 all [installed]
libtirpc-dev/jammy-updates,jammy-security,now 1.3.2-2ubuntu0.1 arm64 [installed,automatic]
libtirpc3/jammy-updates,jammy-security,now 1.3.2-2ubuntu0.1 arm64 [installed]
libtsan0/jammy-updates,jammy-security,now 11.4.0-1ubuntu1~22.04 arm64 [installed,automatic]
libubsan1/jammy-updates,jammy-security,now 12.3.0-1ubuntu1~22.04 arm64 [installed,automatic]
libudev1/jammy-updates,now 249.11-0ubuntu3.12 arm64 [installed]
libunistring2/jammy,now 1.0-1 arm64 [installed]
libuuid1/jammy-updates,jammy-security,now 2.37.2-4ubuntu3.4 arm64 [installed]
libwebp7/jammy-updates,jammy-security,now 1.2.2-2ubuntu0.22.04.2 arm64 [installed,automatic]
libx11-6/jammy-updates,jammy-security,now 2:1.7.5-1ubuntu0.3 arm64 [installed,automatic]
libx11-data/jammy-updates,jammy-security,now 2:1.7.5-1ubuntu0.3 all [installed,automatic]
libxau6/jammy,now 1:1.0.9-1build5 arm64 [installed,automatic]
libxcb1/jammy,now 1.14-3ubuntu3 arm64 [installed,automatic]
libxdmcp6/jammy,now 1:1.1.3-0ubuntu5 arm64 [installed,automatic]
libxext6/jammy,now 2:1.3.4-1build1 arm64 [installed,automatic]
libxml2/jammy-updates,jammy-security,now 2.9.13+dfsg-1ubuntu0.4 arm64 [installed,automatic]
libxmuu1/jammy,now 2:1.1.3-3 arm64 [installed,automatic]
libxpm4/jammy-updates,jammy-security,now 1:3.5.12-1ubuntu0.22.04.2 arm64 [installed,automatic]
libxxhash0/jammy,now 0.8.1-1 arm64 [installed]
libzstd1/jammy,now 1.4.8+dfsg-3build1 arm64 [installed]
linux-libc-dev/jammy-updates,jammy-security,now 5.15.0-117.127 arm64 [installed,automatic]
login/jammy-updates,jammy-security,now 1:4.8.1-2ubuntu2.2 arm64 [installed]
logsave/jammy-updates,jammy-security,now 1.46.5-2ubuntu1.1 arm64 [installed]
lsb-base/jammy,now 11.1.0ubuntu4 all [installed]
make/jammy,now 4.3-4.1build1 arm64 [installed]
manpages-dev/jammy,now 5.10-1ubuntu1 all [installed,automatic]
manpages/jammy,now 5.10-1ubuntu1 all [installed,automatic]
mawk/jammy,now 1.3.4.20200120-3 arm64 [installed]
media-types/jammy,now 7.0.0 all [installed,automatic]
mount/jammy-updates,jammy-security,now 2.37.2-4ubuntu3.4 arm64 [installed]
ncurses-base/jammy-updates,jammy-security,now 6.3-2ubuntu0.1 all [installed]
ncurses-bin/jammy-updates,jammy-security,now 6.3-2ubuntu0.1 arm64 [installed]
netbase/jammy,now 6.3 all [installed,automatic]
openssh-client/jammy-updates,jammy-security,now 1:8.9p1-3ubuntu0.10 arm64 [installed,automatic]
openssl/jammy-updates,jammy-security,now 3.0.2-0ubuntu1.17 arm64 [installed,automatic]
passwd/jammy-updates,jammy-security,now 1:4.8.1-2ubuntu2.2 arm64 [installed]
patch/jammy,now 2.7.6-7build2 arm64 [installed,automatic]
perl-base/jammy-updates,jammy-security,now 5.34.0-3ubuntu1.3 arm64 [installed]
perl-modules-5.34/jammy-updates,jammy-security,now 5.34.0-3ubuntu1.3 all [installed,automatic]
perl/jammy-updates,jammy-security,now 5.34.0-3ubuntu1.3 arm64 [installed,automatic]
pkg-config/jammy,now 0.29.2-1ubuntu3 arm64 [installed]
procps/jammy-updates,jammy-security,now 2:3.3.17-6ubuntu2.1 arm64 [installed]
protobuf-compiler/jammy-updates,jammy-security,now 3.12.4-1ubuntu7.22.04.1 arm64 [installed]
publicsuffix/jammy,now 20211207.1025-1 all [installed,automatic]
python3.10-dev/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed]
python3.10-minimal/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed,automatic]
python3.10/jammy-updates,jammy-security,now 3.10.12-1~22.04.5 arm64 [installed]
readline-common/jammy,now 8.1.2-1 all [installed,automatic]
rpcsvc-proto/jammy,now 1.4.2-0ubuntu6 arm64 [installed,automatic]
sed/jammy,now 4.8-1ubuntu2 arm64 [installed]
sensible-utils/jammy,now 0.0.17 all [installed]
shared-mime-info/jammy,now 2.1-2 arm64 [installed,automatic]
sysvinit-utils/jammy,now 3.01-1ubuntu1 arm64 [installed]
tar/jammy-updates,jammy-security,now 1.34+dfsg-1ubuntu0.1.22.04.2 arm64 [installed]
ubuntu-keyring/jammy,now 2021.03.26 all [installed]
ucf/jammy,now 3.0043 all [installed,automatic]
unzip/jammy-updates,now 6.0-26ubuntu3.2 arm64 [installed]
usrmerge/jammy,now 25ubuntu2 all [installed]
util-linux/jammy-updates,jammy-security,now 2.37.2-4ubuntu3.4 arm64 [installed]
vim-common/jammy-updates,now 2:8.2.3995-1ubuntu2.17 all [installed,automatic]
vim-runtime/jammy-updates,now 2:8.2.3995-1ubuntu2.17 all [installed,automatic]
vim/jammy-updates,now 2:8.2.3995-1ubuntu2.17 arm64 [installed]
wget/jammy-updates,jammy-security,now 1.21.2-2ubuntu1.1 arm64 [installed]
xauth/jammy,now 1:1.1-1build2 arm64 [installed,automatic]
xdg-user-dirs/jammy,now 0.17-2ubuntu4 arm64 [installed,automatic]
xxd/jammy-updates,now 2:8.2.3995-1ubuntu2.17 arm64 [installed,automatic]
xz-utils/jammy,now 5.2.5-2ubuntu1 arm64 [installed,automatic]
zip/jammy,now 3.0-12build2 arm64 [installed]
zlib1g-dev/jammy-updates,jammy-security,now 1:1.2.11.dfsg-2ubuntu9.2 arm64 [installed,automatic]
zlib1g/jammy-updates,jammy-security,now 1:1.2.11.dfsg-2ubuntu9.2 arm64 [installed]
```

# pip list & pip freeze
pip list
```shell
# pip list
Package                                Version         Editable project location
-------------------------------------- --------------- -----------------------------------------------------
absl-py                                2.1.0
attrs                                  23.2.0
backoff                                2.2.1
build                                  1.2.1
CacheControl                           0.14.0
certifi                                2023.7.22
cffi                                   1.16.0
charset-normalizer                     3.2.0
cleo                                   2.1.0
click                                  8.1.7
crashtest                              0.4.1
cryptography                           43.0.0
decorator                              5.1.1
Deprecated                             1.2.14
distlib                                0.3.8
dulwich                                0.21.7
fastjsonschema                         2.20.0
filelock                               3.12.3
fsspec                                 2023.9.0
googleapis-common-protos               1.60.0
grpc-interceptor                       0.15.3
grpcio                                 1.58.0
grpcio-reflection                      1.58.0
grpcio-status                          1.58.0
grpcio-tools                           1.51.1
huggingface-hub                        0.23.0
idna                                   3.4
importlib_metadata                     8.2.0
installer                              0.7.0
jaraco.classes                         3.4.0
jeepney                                0.8.0
Jinja2                                 3.1.2
keyring                                24.3.1
loguru                                 0.6.0
MarkupSafe                             2.1.3
more-itertools                         10.3.0
mpmath                                 1.3.0
msgpack                                1.0.8
mypy-protobuf                          3.4.0
networkx                               3.1
numpy                                  1.26.4
opentelemetry-api                      1.15.0
opentelemetry-exporter-otlp            1.15.0
opentelemetry-exporter-otlp-proto-grpc 1.15.0
opentelemetry-exporter-otlp-proto-http 1.15.0
opentelemetry-instrumentation          0.36b0
opentelemetry-instrumentation-grpc     0.36b0
opentelemetry-proto                    1.15.0
opentelemetry-sdk                      1.15.0
opentelemetry-semantic-conventions     0.36b0
packaging                              23.1
pathlib2                               2.3.7.post1
pexpect                                4.9.0
pip                                    24.2
pkginfo                                1.11.1
platformdirs                           4.2.2
poetry                                 1.8.3
poetry-core                            1.9.0
poetry-plugin-export                   1.8.0
protobuf                               4.24.3
psutil                                 5.9.8
ptyprocess                             0.7.0
pycparser                              2.22
pyproject_hooks                        1.1.0
PyYAML                                 6.0.1
rapidfuzz                              3.9.5
regex                                  2024.7.24
requests                               2.31.0
requests-toolbelt                      1.0.0
safetensors                            0.4.1
scipy                                  1.12.0
SecretStorage                          3.3.3
setuptools                             68.2.0
shellingham                            1.5.4
six                                    1.16.0
sympy                                  1.12
text-embeddings-server                 0.1.0           /TEI/text-embeddings-inference/backends/python/server
tokenizers                             0.19.1
tomli                                  2.0.1
tomlkit                                0.13.0
torch                                  2.1.0
torch-npu                              2.1.0.post6
tqdm                                   4.66.1
transformers                           4.40.2
trove-classifiers                      2024.7.2
typer                                  0.6.1
types-protobuf                         5.27.0.20240626
typing_extensions                      4.7.1
urllib3                                2.0.4
virtualenv                             20.26.3
wheel                                  0.44.0
wrapt                                  1.15.0
zipp                                   3.19.2
```

pip freeze
```shell
# pip freeze
absl-py==2.1.0
attrs==23.2.0
backoff==2.2.1
build==1.2.1
CacheControl==0.14.0
certifi==2023.7.22
cffi==1.16.0
charset-normalizer==3.2.0
cleo==2.1.0
click==8.1.7
crashtest==0.4.1
cryptography==43.0.0
decorator==5.1.1
Deprecated==1.2.14
distlib==0.3.8
dulwich==0.21.7
fastjsonschema==2.20.0
filelock==3.12.3
fsspec==2023.9.0
googleapis-common-protos==1.60.0
grpc-interceptor==0.15.3
grpcio==1.58.0
grpcio-reflection==1.58.0
grpcio-status==1.58.0
grpcio-tools==1.51.1
huggingface-hub==0.23.0
idna==3.4
importlib_metadata==8.2.0
installer==0.7.0
jaraco.classes==3.4.0
jeepney==0.8.0
Jinja2==3.1.2
keyring==24.3.1
loguru==0.6.0
MarkupSafe==2.1.3
more-itertools==10.3.0
mpmath==1.3.0
msgpack==1.0.8
mypy-protobuf==3.4.0
networkx==3.1
numpy==1.26.4
opentelemetry-api==1.15.0
opentelemetry-exporter-otlp==1.15.0
opentelemetry-exporter-otlp-proto-grpc==1.15.0
opentelemetry-exporter-otlp-proto-http==1.15.0
opentelemetry-instrumentation==0.36b0
opentelemetry-instrumentation-grpc==0.36b0
opentelemetry-proto==1.15.0
opentelemetry-sdk==1.15.0
opentelemetry-semantic-conventions==0.36b0
packaging==23.1
pathlib2==2.3.7.post1
pexpect==4.9.0
pkginfo==1.11.1
platformdirs==4.2.2
poetry==1.8.3
poetry-core==1.9.0
poetry-plugin-export==1.8.0
protobuf==4.24.3
psutil==5.9.8
ptyprocess==0.7.0
pycparser==2.22
pyproject_hooks==1.1.0
PyYAML==6.0.1
rapidfuzz==3.9.5
regex==2024.7.24
requests==2.31.0
requests-toolbelt==1.0.0
safetensors==0.4.1
scipy==1.12.0
SecretStorage==3.3.3
shellingham==1.5.4
six==1.16.0
sympy==1.12
-e git+https://github.com/huggingface/text-embeddings-inference.git@cc1c510e8d8af8447c01e6b14c417473cf2dfda9#egg=text_embeddings_server&subdirectory=backends/python/server
tokenizers==0.19.1
tomli==2.0.1
tomlkit==0.13.0
torch==2.1.0
torch-npu==2.1.0.post6
tqdm==4.66.1
transformers==4.40.2
trove-classifiers==2024.7.2
typer==0.6.1
types-protobuf==5.27.0.20240626
typing_extensions==4.7.1
urllib3==2.0.4
virtualenv==20.26.3
wrapt==1.15.0
zipp==3.19.2
```
