IMAGE=dk64:latest
TEST_NAME=test-gym-env
ROM_PATH=/home/ifonlyihadtnt/Programming/_SCHOOL/cs147/DeepKart64/gym-mupen64plus/gym_mupen64plus/ROMs

build:
	docker build -t $(IMAGE) .

verify: build
	docker run -it \
		--name $(TEST_NAME) \
		-p 5900 \
		--mount source="$(ROM_PATH)",target=/src/gym-mupen64plus/gym_mupen64plus/ROMs,type=bind \
		$(IMAGE) \
	# python /src/gym-mupen64plus/verifyEnv.py
	pwd

destroy:
	docker container rm test-gym-env