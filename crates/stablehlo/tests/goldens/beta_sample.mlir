module {
  func.func @sample() -> tensor<f32> {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<3.0> : tensor<f32>
    %2 = stablehlo.constant dense<1.0> : tensor<f32>
    %3 = stablehlo.constant dense<0.0> : tensor<f32>
    %4 = stablehlo.constant dense<1.0> : tensor<f32>
    %5 = stablehlo.compare LT, %0, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = stablehlo.add %0, %4 : tensor<f32>
    %7 = stablehlo.select %5, %6, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %9 = stablehlo.subtract %7, %8 : tensor<f32>
    %10 = stablehlo.constant dense<9.0> : tensor<f32>
    %11 = stablehlo.multiply %10, %9 : tensor<f32>
    %12 = stablehlo.sqrt %11 : tensor<f32>
    %13 = stablehlo.divide %4, %12 : tensor<f32>
    %14 = stablehlo.constant dense<128> : tensor<1xi64>
    %15 = stablehlo.rng %3, %4, %14, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %16 = stablehlo.constant dense<128> : tensor<1xi64>
    %17 = stablehlo.rng %3, %4, %16, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %18 = stablehlo.constant dense<0> : tensor<i32>
    %19 = stablehlo.constant dense<false> : tensor<i1>
    %20 = stablehlo.constant dense<0.0> : tensor<f32>
    %24:3 = stablehlo.while(%21 = %18, %22 = %19, %23 = %20) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %25 = stablehlo.constant dense<128> : tensor<i32>
      %26 = stablehlo.compare LT, %21, %25, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %27 = stablehlo.not %22 : tensor<i1>
      %28 = stablehlo.and %27, %26 : tensor<i1>
      stablehlo.return %28 : tensor<i1>
    } do {
      %29 = stablehlo.dynamic_slice %15, %21, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %30 = stablehlo.reshape %29 : (tensor<1xf32>) -> tensor<f32>
      %31 = stablehlo.dynamic_slice %17, %21, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %32 = stablehlo.reshape %31 : (tensor<1xf32>) -> tensor<f32>
      %33 = stablehlo.multiply %13, %30 : tensor<f32>
      %34 = stablehlo.add %4, %33 : tensor<f32>
      %35 = stablehlo.multiply %34, %34 : tensor<f32>
      %36 = stablehlo.multiply %35, %34 : tensor<f32>
      %37 = stablehlo.multiply %9, %36 : tensor<f32>
      %38 = stablehlo.constant dense<0.5> : tensor<f32>
      %39 = stablehlo.multiply %30, %30 : tensor<f32>
      %40 = stablehlo.multiply %38, %39 : tensor<f32>
      %41 = stablehlo.multiply %9, %36 : tensor<f32>
      %42 = stablehlo.negate %41 : tensor<f32>
      %43 = stablehlo.log %36 : tensor<f32>
      %44 = stablehlo.multiply %9, %43 : tensor<f32>
      %45 = stablehlo.add %40, %9 : tensor<f32>
      %46 = stablehlo.add %45, %42 : tensor<f32>
      %47 = stablehlo.add %46, %44 : tensor<f32>
      %48 = stablehlo.log %32 : tensor<f32>
      %49 = stablehlo.compare LT, %48, %47 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %50 = stablehlo.compare GT, %36, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %51 = stablehlo.and %49, %50 : tensor<i1>
      %52 = stablehlo.constant dense<1> : tensor<i32>
      %53 = stablehlo.add %21, %52 : tensor<i32>
      stablehlo.return %53, %51, %37 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %54 = stablehlo.constant dense<> : tensor<0xi64>
    %55 = stablehlo.rng %3, %4, %54, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %56 = stablehlo.divide %4, %0 : tensor<f32>
    %57 = stablehlo.power %55, %56 : tensor<f32>
    %58 = stablehlo.select %5, %57, %4 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %59 = stablehlo.multiply %24#2, %58 : tensor<f32>
    %60 = stablehlo.divide %59, %2 : tensor<f32>
    %61 = stablehlo.constant dense<0.0> : tensor<f32>
    %62 = stablehlo.constant dense<1.0> : tensor<f32>
    %63 = stablehlo.compare LT, %1, %62 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %64 = stablehlo.add %1, %62 : tensor<f32>
    %65 = stablehlo.select %63, %64, %1 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %66 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %67 = stablehlo.subtract %65, %66 : tensor<f32>
    %68 = stablehlo.constant dense<9.0> : tensor<f32>
    %69 = stablehlo.multiply %68, %67 : tensor<f32>
    %70 = stablehlo.sqrt %69 : tensor<f32>
    %71 = stablehlo.divide %62, %70 : tensor<f32>
    %72 = stablehlo.constant dense<128> : tensor<1xi64>
    %73 = stablehlo.rng %61, %62, %72, distribution = NORMAL : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %74 = stablehlo.constant dense<128> : tensor<1xi64>
    %75 = stablehlo.rng %61, %62, %74, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<128xf32>
    %76 = stablehlo.constant dense<0> : tensor<i32>
    %77 = stablehlo.constant dense<false> : tensor<i1>
    %78 = stablehlo.constant dense<0.0> : tensor<f32>
    %82:3 = stablehlo.while(%79 = %76, %80 = %77, %81 = %78) : tensor<i32>, tensor<i1>, tensor<f32>
    cond {
      %83 = stablehlo.constant dense<128> : tensor<i32>
      %84 = stablehlo.compare LT, %79, %83, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %85 = stablehlo.not %80 : tensor<i1>
      %86 = stablehlo.and %85, %84 : tensor<i1>
      stablehlo.return %86 : tensor<i1>
    } do {
      %87 = stablehlo.dynamic_slice %73, %79, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %88 = stablehlo.reshape %87 : (tensor<1xf32>) -> tensor<f32>
      %89 = stablehlo.dynamic_slice %75, %79, sizes = [1] : (tensor<128xf32>, tensor<i32>) -> tensor<1xf32>
      %90 = stablehlo.reshape %89 : (tensor<1xf32>) -> tensor<f32>
      %91 = stablehlo.multiply %71, %88 : tensor<f32>
      %92 = stablehlo.add %62, %91 : tensor<f32>
      %93 = stablehlo.multiply %92, %92 : tensor<f32>
      %94 = stablehlo.multiply %93, %92 : tensor<f32>
      %95 = stablehlo.multiply %67, %94 : tensor<f32>
      %96 = stablehlo.constant dense<0.5> : tensor<f32>
      %97 = stablehlo.multiply %88, %88 : tensor<f32>
      %98 = stablehlo.multiply %96, %97 : tensor<f32>
      %99 = stablehlo.multiply %67, %94 : tensor<f32>
      %100 = stablehlo.negate %99 : tensor<f32>
      %101 = stablehlo.log %94 : tensor<f32>
      %102 = stablehlo.multiply %67, %101 : tensor<f32>
      %103 = stablehlo.add %98, %67 : tensor<f32>
      %104 = stablehlo.add %103, %100 : tensor<f32>
      %105 = stablehlo.add %104, %102 : tensor<f32>
      %106 = stablehlo.log %90 : tensor<f32>
      %107 = stablehlo.compare LT, %106, %105 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %108 = stablehlo.compare GT, %94, %61 : (tensor<f32>, tensor<f32>) -> tensor<i1>
      %109 = stablehlo.and %107, %108 : tensor<i1>
      %110 = stablehlo.constant dense<1> : tensor<i32>
      %111 = stablehlo.add %79, %110 : tensor<i32>
      stablehlo.return %111, %109, %95 : tensor<i32>, tensor<i1>, tensor<f32>
    }
    %112 = stablehlo.constant dense<> : tensor<0xi64>
    %113 = stablehlo.rng %61, %62, %112, distribution = UNIFORM : (tensor<f32>, tensor<f32>, tensor<0xi64>) -> tensor<f32>
    %114 = stablehlo.divide %62, %1 : tensor<f32>
    %115 = stablehlo.power %113, %114 : tensor<f32>
    %116 = stablehlo.select %63, %115, %62 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %117 = stablehlo.multiply %82#2, %116 : tensor<f32>
    %118 = stablehlo.divide %117, %2 : tensor<f32>
    %119 = stablehlo.add %60, %118 : tensor<f32>
    %120 = stablehlo.divide %60, %119 : tensor<f32>
    return %120 : tensor<f32>
  }
}
