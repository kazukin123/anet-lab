#include "anet/random.hpp"

namespace anet {
	RandomGenerator::RandomGenerator() {
		AutoSeed();
	}

	RandomGenerator& RandomGenerator::Default() {
		static RandomGenerator instance;
		return instance;
	}

	uint64_t RandomGenerator::MakeRandomSeed() {
		std::random_device rd;
		return (static_cast<uint64_t>(rd()) << 32) ^ rd();
	}

	void RandomGenerator::SetSeed(uint64_t seed) {
		seed_ = seed;
		engine_.seed(seed);
		torch::manual_seed(seed);


		if (torch::cuda::is_available()) {
			torch::cuda::manual_seed_all(seed);
		}
	}

	uint64_t RandomGenerator::AutoSeed() {
		uint64_t s = MakeRandomSeed();
		SetSeed(s);
		return s;
	}
}